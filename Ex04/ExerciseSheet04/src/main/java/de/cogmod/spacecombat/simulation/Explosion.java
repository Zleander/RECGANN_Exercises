package de.cogmod.spacecombat.simulation;

import de.cogmod.rgnns.math.Vector3d;

/**
 * @author Sebastian Otte
 */
public class Explosion {
    
    private final Vector3d position;
    private final long starttime;
    
    public Vector3d getPosition() {
        return this.position;
    }
    
    public long getStartTime() {
        return this.starttime;
    }
    
    public Explosion(final Vector3d position, final long starttime) {
        this.position  = position;
        this.starttime = starttime;
    }
}