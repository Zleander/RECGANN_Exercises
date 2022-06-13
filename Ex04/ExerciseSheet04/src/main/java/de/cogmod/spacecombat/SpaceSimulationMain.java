package de.cogmod.spacecombat;

import de.cogmod.spacecombat.simulation.SpaceSimulation;
import de.cogmod.spacecombat.simulation.SpaceSimulationGUI;

/**
 * @author Sebastian Otte
 */
public class SpaceSimulationMain {
    
    public static void main(String[] args) {
        //
        final SpaceSimulation sim       = new SpaceSimulation();
        final AIMComputer aimcontroller = new AIMComputer();
        sim.addObserver(aimcontroller);
        //
        final SpaceSimulationGUI simgui = new SpaceSimulationGUI(
            sim,
            aimcontroller
        );
        //
        simgui.setVisible(true);
        simgui.start();
        //
    } 
    
}