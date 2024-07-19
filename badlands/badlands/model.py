# Copyright 2019 Tristan Salles modified by LX
#
# Badlands is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# Badlands is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Badlands.  If not, see <http://www.gnu.org/licenses/>.

"""
Main components of **badlands** workflow.
"""

import time
import numpy as np
import os
import copy
import datetime
from badlands import elevationTIN, buildMesh
import pandas as pd
import shutil

from scipy.spatial import cKDTree

import io
import os

# check if turn on lake level modification
change_level = 0
if change_level == 1:
    print ('lake level modificaiton is on.')
else:
    print ('lake level modificaiton is off.')

# copy this script first
source = '/home/lxue07/anaconda3/lib/python3.9/site-packages/badlands/model.py'
folder = '/home/lxue07/Documents/badlands/code/test/testlake/modelscript'
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
new_filename = f"backup_model_{timestamp}.py"
shutil.copy2(source, os.path.join(folder, new_filename))

if "READTHEDOCS" not in os.environ:
    from badlands import (
        diffLinear,
        flowNetwork,
        buildMesh,
        waveSed,
        checkPoints,
        buildFlux,
        xmlParser,
        carbGrowth,
        pelagicGrowth,
    )


class Model(object):
    """
    .. image:: img/intro.jpg
       :scale: 50 %
       :alt: capability
       :align: center

    A schematic of **badlands** capabilities illustrating the main variables and forcing conditions. **w** represents the wave boundary conditions,
    **ld** the longshore drift, **sl** the sea-level, **u** the tectonic, **f** the flexural isostasy and **r** the rainfall patterns.
    The stratigraphic evolution and morphology are computed through time.

    Tip:
        If you are not intending to develop new functionalities, the following two functions are the main ways to interact with **badlands** and
        are probably all you need to know about the API documentation |:boom:| |:v:|
    """

    def __init__(self):
        """
        State object for the Badlands model.
        """

        # Simulation state
        self.tNow = 0.0
        self.waveID = 0
        self.outputStep = 0
        self.disp = None
        self.prop = None
        self.carbval = None
        self.carbval2 = None
        self.carbMaxGrowthSp1 = None
        self.carbMaxGrowthSp2 = None
        self.next_carbStep = None
        self.pelaval = None
        self.applyDisp = False
        self.simStarted = False

    def load_xml(self, filename, verbose=False):
        """
        Load the XML input file describing the experiment parameters.

        Args:
            filename : (str) path to the XML file to load.
            verbose : (bool) when :code:`True`, output additional debug information (default: :code:`False`).

        Note:
            * Additional information regarding the input file XML options are found on badlands website_.
            * A good way to start learning **badlands** consists in running the Jupyter Notebooks examples.

        .. _website: https://badlands.readthedocs.io
        """

        np.seterr(divide="ignore", invalid="ignore")

        # Only the first node should create a unique output dir
        self.input = xmlParser.xmlParser(filename, makeUniqueOutputDir=True)
        self.tNow = self.input.tStart

        # Seed the random number generator consistently on all nodes
        seed = None
        # limit to max uint32
        seed = np.random.mtrand.RandomState().tomaxint() % 0xFFFFFFFF
        np.random.seed(seed)

        # If there's no demfile specified, we assume that it will be loaded
        # later using _build_mesh
        if self.input.demfile:
            self._build_mesh(self.input.demfile, verbose)

        # Initialise carbonate evolution if any
        if self.input.carbonate:
            self.carb = carbGrowth.carbGrowth(
                self.input, self.recGrid.regX, self.recGrid.regY, self.carbTIN.tinBase
            )
            if self.input.coastdist > 0.0:
                self.carb.buildReg(self.FVmesh.node_coords[:, :2])

        # Initialise pelagic evolution if any
        if self.input.pelagic:
            self.pelagic = pelagicGrowth.pelagicGrowth(self.input)

        # Initialize TIN mobilized wave sediments going to fluvial flowNetwork
        # self.waveMobile = np.zeros(self.totPts, dtype=float)
        # self.waveED = np.zeros(self.totPts, dtype=float)

    def _build_mesh(self, filename, verbose):
        """
        Build TIN based on regular grid.
        """

        # Construct Badlands mesh and grid to run simulation
        (
            self.recGrid,
            self.FVmesh,
            self.force,
            self.lGIDs,
            self.fixIDs,
            self.inIDs,
            parentIDs,
            self.inGIDs,
            self.totPts,
            self.elevation,
            self.cumdiff,
            self.cumhill,
            self.cumfail,
            self.cumflex,
            self.strata,
            self.mapero,
            self.tinFlex,
            self.flex,
            self.wave,
            self.straTIN,
            self.carbTIN,
        ) = buildMesh.construct_mesh(self.input, filename, verbose)

        if self.input.waveSed:
            self.wavediff = np.zeros((self.totPts))
        else:
            self.wavediff = None

        # Initialize TIN slope (gradient)
        self.slopeTIN = np.zeros(self.totPts, dtype=float)

        # Define hillslope parameters
        self.rain = np.zeros(self.totPts, dtype=float)
        self.hillslope = diffLinear()
        self.hillslope.CDaerial = self.input.CDa
        self.hillslope.CDmarine = self.input.CDm
        self.hillslope.CDriver = self.input.CDr
        self.hillslope.Sfail = self.input.Sfail
        self.hillslope.Cfail = self.input.Cfail
        self.hillslope.Sc = self.input.Sc
        self.hillslope.updatedt = 0

        # Define flow parameters
        self.flow = flowNetwork(self.input)

        if self.input.erolays is None:
            self.flow.erodibility = np.full(self.totPts, self.input.SPLero)
        else:
            self.flow.erodibility = self.mapero.erodibility
        self.flow.mindt = self.input.minDT
        self.flow.xycoords = self.FVmesh.node_coords[:, :2]
        self.flow.spl = self.input.spl
        self.flow.depo = self.input.depo
        self.flow.xgrid = None

        reassignID = np.where(parentIDs < len(parentIDs))[0]
        if len(reassignID) > 0:
            tmpTree = cKDTree(self.flow.xycoords[len(parentIDs) :, :2])
            distances, indices = tmpTree.query(self.flow.xycoords[reassignID, :2], k=1)
            indices += len(parentIDs)
            parentIDs[reassignID] = indices

        self.flow.parentIDs = parentIDs

        # Define hydrodynamic conditions
        if self.input.waveOn:
            self.force.next_wave = self.input.tStart
            self.wave.build_tree(self.FVmesh.node_coords[:, :2])
            self.wave.swan_init(
                self.input, self.elevation, self.waveID, self.force.sealevel
            )
        else:
            if self.input.waveSed:
                self.force.next_wave = self.input.tStart + self.input.tWave
            else:
                self.force.next_wave = self.input.tEnd + 1.0e5

        if self.input.carb:

            self.next_carbStep = self.input.tStart + self.input.tCarb
            self.oldsed = np.zeros(len(self.elevation))
            if self.carbTIN is not None:
                self.prop = np.zeros((self.totPts, self.carbTIN.nbSed))
        else:
            self.next_carbStep = self.input.tEnd + 1.0e5
            self.prop = np.zeros((self.totPts, 1))

    def _rebuild_mesh(self, verbose=False):
        """
        Build TIN after 3D displacements.
        """

        # Build the Finite Volume representation
        self.fixIDs = self.recGrid.boundsPt + self.recGrid.edgesPt
        (
            self.FVmesh,
            self.lGIDs,
            self.inIDs,
            self.inGIDs,
            self.totPts,
        ) = buildMesh.reconstruct_mesh(self.recGrid, self.input, verbose)

        # Update edges elevation
        tree1 = cKDTree(self.FVmesh.node_coords[self.fixIDs :, :2])
        tmpelev = self.elevation[self.fixIDs :]
        distances, indices = tree1.query(
            self.FVmesh.node_coords[: self.fixIDs, :2], k=1
        )
        self.elevation[: self.fixIDs] = tmpelev[indices]
        self.hillslope.ids = None

        # Reset TIN kdtree and rain
        self.force.update_force_TIN(self.FVmesh.node_coords[:, :2])
        self.rain = np.zeros(self.totPts, dtype=float)
        self.rain[self.inIDs] = self.force.get_Rain(
            self.tNow, self.elevation, self.inIDs
        )

        # Update flexural isostasy
        if self.input.flexure:
            self.tinFlex = np.zeros(self.totPts, dtype=float)
            self.flex.update_flexure_parameters(self.FVmesh.node_coords[:, :2])

        # Update SWAN mesh
        if self.input.waveOn:
            self.wave.build_tree(self.FVmesh.node_coords[:, :2])

        # Update stratigraphic mesh
        if self.input.stratdx > 0:
            self.strata.update_TIN(self.FVmesh.node_coords[:, :2])

        # Update erodibility maps
        if self.input.erolays is None:
            self.flow.erodibility = np.full(self.totPts, self.input.SPLero)
        else:
            self.flow.erodibility = self.mapero.erodibility

        # Update Wavesed grid interpolation
        if self.input.waveSed:
            wave.build_tree(self.FVmesh.node_coords[:, :2])

        # Update Carbonate mesh
        # if self.input.carbonate:
        #    if self.input.coastdist>0.:
        #        self.carb.buildReg(self.FVmesh.node_coords[:,:2])

        self.flow.xycoords = self.FVmesh.node_coords[:, :2]
        self.flow.xgrid = None
        self.flow.sedload = None
        self.flow.flowdensity = None
        self.flow.domain = None
        self.hillslope.updatedt = 0

        self.carbval = None
        self.carbval2 = None
        self.pelaval = None
        self.prop = np.zeros((self.totPts, 1))

    def run_to_time(self, tEnd, verbose=False):
        """
        Run the simulation to a specified point in time.

        Args:
            tEnd : (float) time in years up to run the model for...
            verbose : (bool) when :code:`True`, output additional debug information (default: :code:`False`).

        Warning:
            If specified end time (**tEnd**) is greater than the one defined in the XML input file priority
            is given to the XML value.
        """

        assert hasattr(
            self, "recGrid"
        ), "DEM file has not been loaded. Configure one in your XML file or call the build_mesh function."

        if tEnd > self.input.tEnd:
            print(
                "Specified end time is greater than the one used in the XML input file and has been adjusted!"
            )
            tEnd = self.input.tEnd

        # Define non-flow related processes times
        if not self.simStarted:
            self.force.next_rain = self.force.T_rain[0, 0]
            self.force.next_disp = self.force.T_disp[0, 0]
            self.force.next_carb = self.force.T_carb[0, 0]

            self.force.next_display = self.input.tStart
            if self.input.laytime > 0:
                self.force.next_layer = self.input.tStart + self.input.laytime
            else:
                self.force.next_layer = self.input.tEnd + 1000.0
            self.exitTime = self.input.tEnd
            if self.input.flexure:
                self.force.next_flexure = self.input.tStart + self.input.ftime
            else:
                self.force.next_flexure = self.exitTime + self.input.tDisplay
            self.simStarted = True

        outStrata = 0
        last_time = time.process_time()
        last_output = time.process_time()

        # Perform main simulation loop
        while self.tNow < tEnd:


            # At most, display output every 5 seconds
            
            tloop = time.process_time() - last_time
            #print("t2d, tNow = %s (step took %0.02f seconds)" % (self.tNow, tloop)) # check tNow XL
            if time.process_time() - last_output >= 5.0:
                print("t1, tNow = %s (step took %0.02f seconds)" % (self.tNow, tloop))
                last_output = time.process_time()
            last_time = time.process_time()

            # Load precipitation rate
            if (
                self.force.next_rain <= self.tNow
                and self.force.next_rain < self.input.tEnd
            ):
                if self.tNow == self.input.tStart:
                    ref_elev = buildMesh.get_reference_elevation(
                        self.input, self.recGrid, self.elevation
                    )
                    self.force.getSea(self.tNow, self.input.udw, ref_elev)
                self.rain = np.zeros(self.totPts, dtype=float)
                self.rain[self.inIDs] = self.force.get_Rain(
                    self.tNow, self.elevation, self.inIDs
                )

            # Initialize waveFlux at tStart
            # if self.tNow == self.input.tStart:
            #     self.force.initWaveFlux(self.inIDs)

            # Load tectonic grid
            if not self.input.disp3d:
                # Vertical displacements
                if (
                    self.force.next_disp <= self.tNow
                    and self.force.next_disp < self.input.tEnd
                ):
                    ldisp = np.zeros(self.totPts, dtype=float)
                    ldisp.fill(-1.0e6)
                    # load_Tecto_map: perform interpolation from regular grid to TIN one at this time interval - XL
                    ldisp[self.inIDs] = self.force.load_Tecto_map(self.tNow, self.inIDs) 
                    self.disp = self.force.disp_border(
                        ldisp,
                        self.FVmesh.neighbours,
                        self.FVmesh.edge_length,
                        self.recGrid.boundsPt,
                    )
                    self.applyDisp = True
            else:
                # 3D displacements
                if (
                    self.force.next_disp <= self.tNow
                    and self.force.next_disp < self.input.tEnd
                ):
                    if self.input.laytime == 0:
                        updateMesh = self.force.load_Disp_map(
                            self.tNow, self.FVmesh.node_coords[:, :2], self.inIDs
                        )
                    else:
                        # Define 3D displacements on the stratal regions
                        if self.strata is not None:
                            updateMesh, regdX, regdY = self.force.load_Disp_map(
                                self.tNow,
                                self.FVmesh.node_coords[:, :2],
                                self.inIDs,
                                True,
                                self.strata.xyi,
                                self.strata.ids,
                            )
                        else:
                            updateMesh = self.force.load_Disp_map(
                                self.tNow, self.FVmesh.node_coords[:, :2], self.inIDs
                            )

                    # Update mesh when a 3D displacements field has been loaded
                    if updateMesh:
                        self.force.dispZ = self.force.disp_border(
                            self.force.dispZ,
                            self.FVmesh.neighbours,
                            self.FVmesh.edge_length,
                            self.recGrid.boundsPt,
                        )
                        # Define flexural flags
                        fflex = 0
                        flexiso = None
                        if self.input.flexure:
                            flexiso = self.cumflex
                            fflex = 1
                        # Define stratal flags
                        fstrat = 0
                        sload = None
                        if (
                            self.input.udw == 1
                            and self.tNow == self.input.tStart
                            and self.strata is not None
                        ):
                            if self.strata.oldload is None:
                                self.strata.oldload = np.zeros(
                                    len(self.elevation), dtype=float
                                )
                        if self.strata is not None:
                            if self.strata.oldload is None:
                                self.strata.oldload = np.zeros(
                                    len(self.elevation), dtype=float
                                )
                        if self.input.laytime > 0 and self.strata.oldload is not None:
                            sload = self.strata.oldload
                            fstrat = 1
                        # Define erodibility map flags
                        fero = 0
                        vKe = None
                        vTh = None
                        if self.input.erolays is not None:
                            if self.input.erolays >= 0:
                                fero = 1
                                vKe = self.mapero.Ke
                                vTh = self.mapero.thickness
                        # Apply horizontal displacements
                        (
                            self.recGrid.tinMesh,
                            self.elevation,
                            self.cumdiff,
                            self.cumhill,
                            self.cumfail,
                            self.wavediff,
                            fcum,
                            scum,
                            Ke,
                            Th,
                        ) = self.force.apply_XY_displacements(
                            self.recGrid.areaDel,
                            self.fixIDs,
                            self.elevation,
                            self.cumdiff,
                            self.cumhill,
                            self.cumfail,
                            self.wavediff,
                            tflex=flexiso,
                            scum=sload,
                            Te=vTh,
                            Ke=vKe,
                            flexure=fflex,
                            strat=fstrat,
                            ero=fero,
                        )
                        # Update relevant parameters in deformed TIN
                        if fflex == 1:
                            self.cumflex = fcum
                        if fero == 1:
                            self.mapero.Ke = Ke
                            self.mapero.thickness = Th
                        # Rebuild the computational mesh, also 
                        # recalculate the pit and using lake maxfill -XL
                        self._rebuild_mesh(verbose)
                        print ('check is rebuild_mesh is used here')

                        # In case where the paleoflow workflow is used
                        if self.force.uDisp is not None:
                            self.elevation += self.force.uDisp

                        # Update the stratigraphic mesh
                        if self.input.laytime > 0 and self.strata is not None:
                            self.strata.move_mesh(regdX, regdY, scum, verbose)

            # Compute isostatic flexure
            if self.tNow >= self.force.next_flexure:
                flextime = time.process_time()
                ref_elev = buildMesh.get_reference_elevation(
                    self.input, self.recGrid, self.elevation
                )
                self.force.getSea(self.tNow, self.input.udw, ref_elev)
                self.tinFlex = self.flex.get_flexure(
                    self.elevation,
                    self.cumdiff,
                    self.force.sealevel,
                    self.recGrid.boundsPt,
                    initFlex=False,
                )
                # Get border values
                self.tinFlex = self.force.disp_border(
                    self.tinFlex,
                    self.FVmesh.neighbours,
                    self.FVmesh.edge_length,
                    self.recGrid.boundsPt,
                )
                # Update flexural parameters
                self.elevation += self.tinFlex
                self.cumflex += self.tinFlex
                # Update next flexure time
                self.force.next_flexure += self.input.ftime
                print(
                    "   - Compute flexural isostasy %0.02f seconds"
                    % (time.process_time() - flextime)
                )

            # Compute wavesed parameters
            if self.tNow >= self.force.next_wave:
                wavetime = time.process_time()
                if self.carbTIN is not None:
                    # Update erosion/deposition due to SPM processes on carbTIN
                    self.carbTIN.update_layers(
                        self.cumdiff - self.oldsed, self.elevation
                    )
                    self.carbTIN.get_active_layer(self.input.tWave * self.input.wEro)
                    actlay = self.carbTIN.alay
                else:
                    actlay = None
                # Compute wave field and associated bottom current conditions
                waveED, nactlay = self.wave.compute_wavesed(
                    self.tNow, self.input, self.force, self.elevation, actlay
                )
                # Wave-remobilized sediments sent to stream network if mobilized over steep slopes
                # slopeVal = 0.01
                # slopeBool = (self.slopeTIN > slopeVal).astype(int)
                # waveDep = waveED.clip(min=0)  # keep positive values (deposition)
                # self.waveMobile = np.multiply(slopeBool, waveDep)
                # self.waveED = np.subtract(waveED, self.waveMobile)
                # self.force.waveFlux = (
                #     np.multiply(self.waveMobile, self.FVmesh.control_volumes)
                #     / self.input.tWave
                # )

                # Update elevation / cumulative changes based on wave-induced sediment transport
                self.elevation += waveED
                self.cumdiff += waveED
                self.wavediff += waveED
                self.oldsed = np.copy(self.cumdiff)
                # self.elevation += self.waveED
                # self.cumdiff += self.waveED
                # self.wavediff += self.waveED
                print(
                    "   - Compute wave-induced sediment transport %0.02f seconds"
                    % (time.process_time() - wavetime)
                )
                # Update carbonate active layer
                if nactlay is not None:
                    self.carbTIN.update_active_layer(nactlay, self.elevation)
                # Update next wave time step
                self.force.next_wave += self.input.tWave

            # Compute carbonate evolution
            if self.tNow >= self.next_carbStep:
                carbtime = time.process_time()
                depth = self.elevation - self.force.sealevel
                if self.carbTIN is not None:
                    # Update erosion/deposition due to river and diffusion on carbTIN
                    self.carbTIN.update_layers(
                        self.cumdiff - self.oldsed, self.elevation
                    )

                # Compute reef growth
                if self.input.carbonate:

                    # Load carbonate growth rates for species 1 and 2 during a given growth event
                    if (
                        self.force.next_carb <= self.tNow
                        and self.force.next_carb < self.input.tEnd
                    ):
                        (
                            self.carbMaxGrowthSp1,
                            self.carbMaxGrowthSp2,
                        ) = self.force.get_carbGrowth(self.tNow, self.inIDs)
                    self.carbval, self.carbval2 = self.carb.computeCarbonate(
                        self.force.meanH,
                        self.cumdiff - self.oldsed,
                        depth,
                        self.carbMaxGrowthSp1,
                        self.carbMaxGrowthSp2,
                        self.input.tCarb,
                    )

                    if self.carbval2 is not None:
                        self.cumdiff += self.carbval + self.carbval2
                        self.elevation += self.carbval + self.carbval2
                    else:
                        self.cumdiff += self.carbval
                        self.elevation += self.carbval
                    if self.carbTIN is not None:
                        self.carbTIN.paleoDepth[:, self.carbTIN.step] = self.elevation
                        self.carbTIN.depoThick[:, self.carbTIN.step, 1] += self.carbval
                        self.carbTIN.layerThick[:, self.carbTIN.step] += self.carbval
                        if self.carbval2 is not None:
                            self.carbTIN.depoThick[
                                :, self.carbTIN.step, 2
                            ] += self.carbval2
                            self.carbTIN.layerThick[
                                :, self.carbTIN.step
                            ] += self.carbval2
                # Compute pelagic rain
                if self.input.pelagic:
                    self.pelaval = self.pelagic.computePelagic(depth, self.input.tCarb)
                    self.cumdiff += self.pelaval
                    self.elevation += self.pelaval
                    if self.carbTIN is not None:
                        self.carbTIN.paleoDepth[:, self.carbTIN.step] = self.elevation
                        self.carbTIN.depoThick[:, self.carbTIN.step, 0] += self.pelaval
                        self.carbTIN.layerThick[:, self.carbTIN.step] += self.pelaval
                # Update proportion based on top layer
                if self.prop is not None:
                    ids = np.where(self.carbTIN.layerThick[:, self.carbTIN.step] > 0.0)[
                        0
                    ]
                    self.prop.fill(0.0)
                    self.prop[ids, 0] = (
                        self.carbTIN.depoThick[ids, self.carbTIN.step, 0]
                        / self.carbTIN.layerThick[ids, self.carbTIN.step]
                    )
                    if self.input.carbonate:
                        self.prop[ids, 1] = (
                            self.carbTIN.depoThick[ids, self.carbTIN.step, 1]
                            / self.carbTIN.layerThick[ids, self.carbTIN.step]
                        )
                        if self.carbval2 is not None:
                            self.prop[ids, 2] = (
                                self.carbTIN.depoThick[ids, self.carbTIN.step, 2]
                                / self.carbTIN.layerThick[ids, self.carbTIN.step]
                            )

                # Update current cumulative erosion deposition
                self.oldsed = np.copy(self.cumdiff)
                self.next_carbStep += self.input.tCarb
                print(
                    "   - Compute carbonate growth %0.02f seconds"
                    % (time.process_time() - carbtime)
                )

            # Update next stratal layer time
            if self.tNow >= self.force.next_layer:
                self.force.next_layer += self.input.laytime
                if self.straTIN is not None:
                    self.straTIN.step += 1
                if self.input.laststrat == True:
                   outStrata=0
                if self.strata:
                    if self.tNow==tEnd: 
                        self.write=1 # set parameter to call hdf5 stratal writer on final strat only
                    else:
                        self.write=0
                    if self.input.laststrat == False:
                        self.write=outStrata #revert to previous behaviour by default
                    sub = self.strata.buildStrata(
                        self.elevation,
                        self.cumdiff,
                        self.force.sealevel,
                        self.recGrid.boundsPt,
                        self.write,
                        self.outputStep,
                    )
                    self.elevation += sub
                    self.cumdiff += sub
                outStrata = 0

            # if self.tNow >0.4e6 and self.tNow < 0.6e6:
            #     df = pd.DataFrame()
            #     df['x'] = self.FVmesh.node_coords[:, 0]
            #     df['y'] = self.FVmesh.node_coords[:, 1]
            #     df['fillH'] = self.fillH
            #     df['ele'] = self.elevation
            #     df['sed'] = self.cumdiff
            #     df.to_csv('testoutput/beforefill_{}.csv'.format(self.tNow))

            # Compute stream network  # where the lake depth + elevation is first generated - XL
            # check if change the lake level 
             
            if change_level == 1:
                print ('lake depth: ', int(self.input.fillmax), ' m')
                #curve = pd.read_csv('data/lake_level_curve_1my_twophase_rise.csv')
                curve = pd.read_csv('data/lake_level_curve_1my.csv')
                hist = curve['hist'].values
                llchange = curve['llchange'].values
                # find out lake level change in this step
                ll_id = [i for i, num in enumerate(hist) if int(num) <= self.tNow][-1]
                self.input.fillmax = llchange[ll_id]
                elevationTIN.assign_parameter_pit(
                self.FVmesh.neighbours,
                self.FVmesh.control_volumes,
                self.input.diffnb,
                self.input.diffprop,
                self.input.propa,
                self.input.propb,
                self.recGrid.boundsPt,
                self.input.fillmax,
                )

            # regular stream flow function
            self.fillH, self.elevation = buildFlux.streamflow(
                self.input,
                self.FVmesh,
                self.recGrid,
                self.force,
                self.hillslope,
                self.flow,
                self.elevation,
                self.lGIDs,
                self.rain,
                self.tNow,
                verbose,
            )
            
            # if lake change by extral function
            #lake_change=0
            # if lake_change ==1:
            #     # #replace fillH with customized lake level data (method2 using output fillH)
            #     # period = 1e6 # yr
            #     # llmax = 400 #100 # m lake change maxmium 
            #     # #self.input.fillmax = 20
            #     # time_max = 1e6 # year
            #     # n_step = 100
            #     # # time
            #     # hist = np.linspace(0, time_max, n_step, dtype = int)
            #     # # lake level changes
            #     # llchange = 0.5*llmax*np.sin(2*np.pi*hist/period- 0.5*np.pi) + 0.5*llmax  # only rise
            #     # #llchange = 0.5*llmax*np.sin(2*np.pi*hist/period) # sin curve
            #     # #llchange = -0.5*llmax*np.sin(2*np.pi*hist/period- 0.5*np.pi) -0.5*llmax # only drop 
            #     # read the lake level data
            #     curve = pd.read_csv('data/lake_level_curve_1my.csv')
            #     #curve = pd.read_csv('data/lake_level_curve_1my_twophase_rise.csv')
            #     hist = curve['hist'].values
            #     llchange = curve['llchange'].values
            #     df = pd.DataFrame()
            #     # calculate lake thickness, only change regions where lake depth > 0
            #     df['lake_depth'] = self.fillH - self.elevation
            #     # find out lake level change in this step
            #     ll_id = [i for i, num in enumerate(hist) if int(num) <= self.tNow][-1]
            #     # print ('step ', ll_id, self.tNow, llchange[ll_id])
            #     df['now_ll_change'] = 0 # make a new column
            #     df.loc[df['lake_depth']>1, 'now_ll_change'] =  llchange[ll_id]
            #     #df.to_csv('testoutput/beforefill_{}.csv'.format(self.tNow))

            #     # add the rised lake level  
            #     self.fillH = self.elevation + df['now_ll_change'].values
            #     #self.fillH = self.fillH + df['now_ll_change'].values
            #     # avoid  fillh bneath topograph
            #     self.fillH = np.maximum(self.fillH,self.elevation)
            #     # avoid fillh higher than topograph
            #     # elemax = np.max(self.elevation)
            #     # self.fillH = np.where(self.fillH>elemax, elemax, self.fillH)

            #     # now try different functions in streamflow
            #     fillH_fix = self.fillH

            #     if self.tNow >0.4e6 and self.tNow < 0.6e6:
            #         df = pd.DataFrame()
            #         df['x'] = self.FVmesh.node_coords[:, 0]
            #         df['y'] = self.FVmesh.node_coords[:, 1]
            #         df['fillH'] = self.fillH
            #         df['ele'] = self.elevation
            #         df['sed'] = self.cumdiff
            #         df.to_csv('testoutput/firstfill_{}.csv'.format(self.tNow))

            #     # copy some information
            #     testclass = copy.deepcopy(self)

            #     # #################################################
            #     # if self.input.tStart == self.tNow and self.input.nopit == 1:
            #     #     self.fillH = self.elevationTIN.pit_stack(self.elevation, self.input.nopit, self.force.sealevel)
            #     #     self.elevation = self.fillH
            #     # else:
            #     #     self.fillH = elevationTIN.pit_stack(fillH_fix, 0, self.force.sealevel)
            #     # self.fillH = elevationTIN.pit_stack(fillH_fix, 0, self.force.sealevel)

            #     # self.flow.SFD_receivers(self.fillH, fillH_fix, self.FVmesh.neighbours, 
            #     #     self.FVmesh.vor_edges, self.FVmesh.edge_length, self.lGIDs)
            #     # self.flow.maxh = 10*self.flow.maxh
            #     # self.flow.maxdep = 10*self.flow.maxdep
            #     # self.flow.localbase = self.flow.base
            #     # self.flow.ordered_node_array_filled()
            #     # self.flow.stack = self.flow.localstack
            #     # self.flow.localbase1 = self.flow.base1
            #     # self.flow.ordered_node_array_elev()
            #     # self.flow.stack1 = self.flow.localstack1
            #     # self.flow.compute_parameters_depression(
            #     #    self.fillH, fillH_fix, self.FVmesh.control_volumes, 
            #     #    self.force.sealevel)
            #     # we need a loop to fully fill the depression of fillh
            #     # when lake rise, there should be filling on the gap
            #     # when lake drop there should be no fills
            #     # 10 loops could be fine for a test of lake level rise of 200
            #     #print (llchange[ll_id])
            
            #     if llchange[ll_id]>1e-3:
            #         for i in range(10):
            #             self.fillH, self.elevation = buildFlux.streamflow(
            #                 self.input,
            #                 self.FVmesh,
            #                 self.recGrid,
            #                 self.force,
            #                 self.hillslope,
            #                 self.flow,
            #                 self.fillH,
            #                 self.lGIDs,
            #                 self.rain,
            #                 self.tNow,
            #                 verbose,
            #             )
            #     else:
            #         #self.input.fillmax = 1
            #         self.fillH, self.elevation = buildFlux.streamflow(
            #         self.input,
            #         self.FVmesh,
            #         self.recGrid,
            #         self.force,
            #         self.hillslope,
            #         self.flow,
            #         self.fillH,
            #         self.lGIDs,
            #         self.rain,
            #         self.tNow,
            #         verbose,
            #         )
            #     print (self.input.fillmax)   
            #     new_fill = self.fillH
                
            #     # test deep copy 
            #     # self.elevation += 2000
            #     ##############################################################

            #     if self.tNow >0.4e6 and self.tNow < 0.6e6:
            #         df = pd.DataFrame()
            #         df['x'] = self.FVmesh.node_coords[:, 0]
            #         df['y'] = self.FVmesh.node_coords[:, 1]
            #         df['fillH'] = self.fillH
            #         df['ele'] = self.elevation
            #         df['sed'] = self.cumdiff
            #         df.to_csv('testoutput/sefill_{}.csv'.format(self.tNow))

            #     # copy back 
            #     self = copy.deepcopy(testclass)
            #     self.fillH = new_fill
            #     if self.tNow >0.4e6 and self.tNow < 0.6e6:
            #         df = pd.DataFrame()
            #         df['x'] = self.FVmesh.node_coords[:, 0]
            #         df['y'] = self.FVmesh.node_coords[:, 1]
            #         df['fillH'] = self.fillH
            #         df['ele'] = self.elevation
            #         df['sed'] = self.cumdiff
            #         df.to_csv('testoutput/thirdfill_{}.csv'.format(self.tNow))


            # Create checkpoint files and write HDF5 output
            if self.tNow >= self.force.next_display:
                if self.force.next_display > self.input.tStart:
                    outStrata = 1
                checkPoints.write_checkpoints(
                    self.input,
                    self.recGrid,
                    self.lGIDs,
                    self.inIDs,
                    self.tNow,
                    self.FVmesh,
                    self.force,
                    self.flow,
                    self.rain,
                    self.elevation,
                    self.fillH,
                    self.cumdiff,
                    self.cumhill,
                    self.cumfail,
                    self.wavediff,
                    self.outputStep,
                    self.prop,
                    self.mapero,
                    self.cumflex,
                )

                if self.straTIN is not None and self.outputStep % self.input.tmesh == 0:
                    meshtime = time.process_time()
                    self.straTIN.write_hdf5_stratigraphy(self.lGIDs, self.outputStep)
                    print(
                        "   - Write sediment mesh output %0.02f seconds"
                        % (time.process_time() - meshtime)
                    )

                if self.carbTIN is not None and self.outputStep % self.input.tmesh == 0:
                    meshtime = time.process_time()
                    self.carbTIN.write_hdf5_stratigraphy(self.lGIDs, self.outputStep)
                    print(
                        "   - Write carbonate mesh output %0.02f seconds"
                        % (time.process_time() - meshtime)
                    )

                # Update next display time
                last_output = time.process_time()
                self.force.next_display += self.input.tDisplay
                self.outputStep += 1
                if self.carbTIN is not None:
                    self.carbTIN.step += 1

            # Get the maximum time before updating one of the above processes / components
            tStop = min(
                [
                    self.force.next_display,
                    self.force.next_layer,
                    self.force.next_flexure,
                    tEnd,
                    self.force.next_wave,
                    self.force.next_disp,
                    self.force.next_rain,
                    self.next_carbStep,
                ]
            )

            #print("t2b, tNow = %s (step took %0.02f seconds)" % (self.tNow, tloop)) # check tNow XL
            if tStop < tEnd:
                (
                    self.tNow,
                    self.elevation,
                    self.cumdiff,
                    self.cumhill,
                    self.cumfail,
                    self.slopeTIN,
                ) = buildFlux.sediment_flux(    # this si where the tNow is updated as minDT in the loop XL 
                    self.input,
                    self.recGrid,
                    self.hillslope,
                    self.FVmesh,
                    self.flow,
                    self.force,
                    self.rain,
                    self.lGIDs,
                    self.applyDisp,
                    self.straTIN,
                    self.mapero,
                    self.cumdiff,
                    self.cumhill,
                    self.cumfail,
                    self.fillH,
                    self.disp,
                    self.inGIDs,
                    self.elevation,
                    self.tNow,
                    tStop,
                    verbose,
                )
                #print("t2e, tNow = %s (step took %0.02f seconds)" % (self.tNow, tloop)) # check tNow XL
            else:
                self.tNow = tEnd
                #print("t2c, tNow = %s (step took %0.02f seconds)" % (self.tNow, tloop)) # check tNow XL

            # output lake for the final step
            # df = pd.DataFrame()
            # df['x'] = self.FVmesh.node_coords[:, 0]
            # df['y'] = self.FVmesh.node_coords[:, 1]
            # df['fillH'] = self.fillH
            # df['ele'] = self.elevation
            # df.to_csv('data_{}.csv'.format(self.tNow))




        tloop = time.process_time() - last_time
        print("t3, tNow = %s (%0.02f seconds)" % (self.tNow, tloop))

        # Isostatic flexure
        if self.input.flexure:
            flextime = time.process_time()
            ref_elev = buildMesh.get_reference_elevation(
                self.input, self.recGrid, self.elevation
            )
            self.force.getSea(self.tNow, self.input.udw, ref_elev)
            self.tinFlex = self.flex.get_flexure(
                self.elevation,
                self.cumdiff,
                self.force.sealevel,
                self.recGrid.boundsPt,
                initFlex=False,
            )
            # Get border values
            self.tinFlex = self.force.disp_border(
                self.tinFlex,
                self.FVmesh.neighbours,
                self.FVmesh.edge_length,
                self.recGrid.boundsPt,
            )
            # Update flexural parameters
            self.elevation += self.tinFlex
            self.cumflex += self.tinFlex
            # Update next flexure time
            self.force.next_flexure += self.input.ftime
            print(
                "   - Compute flexural isostasy %0.02f seconds"
                % (time.process_time() - flextime)
            )

        # Update next stratal layer time
        if self.tNow >= self.force.next_layer:
            self.force.next_layer += self.input.laytime
            if self.input.laststrat==True: 
                self.write=1 # set parameter to call hdf5 stratal writer
            sub = self.strata.buildStrata(
                self.elevation,
                self.cumdiff,
                self.force.sealevel,
                self.recGrid.boundsPt,
                self.write, #was 0
                self.outputStep + 1,
            )
            self.elevation += sub
            self.cumdiff += sub
            
        # Create checkpoint files and write HDF5 output
        if (
            self.input.udw == 0
            or self.tNow == self.input.tEnd
            or self.tNow == self.force.next_display
        ):
            checkPoints.write_checkpoints(
                self.input,
                self.recGrid,
                self.lGIDs,
                self.inIDs,
                self.tNow,
                self.FVmesh,
                self.force,
                self.flow,
                self.rain,
                self.elevation,
                self.fillH,
                self.cumdiff,
                self.cumhill,
                self.cumfail,
                self.wavediff,
                self.outputStep,
                self.prop,
                self.mapero,
                self.cumflex,
            )

            if self.straTIN is not None and self.outputStep % self.input.tmesh == 0:
                meshtime = time.process_time()
                self.straTIN.write_hdf5_stratigraphy(self.lGIDs, self.outputStep)
                print(
                    "   - Write sediment mesh output %0.02f seconds"
                    % (time.process_time() - meshtime)
                )

            if self.carbTIN is not None and self.outputStep % self.input.tmesh == 0:
                meshtime = time.process_time()
                self.carbTIN.write_hdf5_stratigraphy(self.lGIDs, self.outputStep)
                print(
                    "   - Write carbonate mesh output %0.02f seconds"
                    % (time.process_time() - meshtime)
                )

            self.force.next_display += self.input.tDisplay
            self.outputStep += 1
            if self.straTIN is not None:
                self.straTIN.write_hdf5_stratigraphy(self.lGIDs, self.outputStep - 1)
            if self.carbTIN is not None:
                self.carbTIN.write_hdf5_stratigraphy(self.lGIDs, self.outputStep - 1)
                self.carbTIN.step += 1

