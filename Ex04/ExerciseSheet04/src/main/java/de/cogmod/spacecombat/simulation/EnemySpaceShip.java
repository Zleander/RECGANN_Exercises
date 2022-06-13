package de.cogmod.spacecombat.simulation;


import java.util.Random;

import de.cogmod.rgnns.math.Vector3d;

/**
 * @author Sebastian Otte
 */
public class EnemySpaceShip {
    
    private Vector3d origin      = new Vector3d();
    private Vector3d relposition = new Vector3d();
    private Vector3d position    = new Vector3d();
    private long     t;
    private boolean  destroyed   = false;
    
    private final double[] frequenciesx = {0.15, 0.311, 0.43};
    private final double[] frequenciesy = {0.21, 0.29,  0.5};
    private final double[] frequenciesz = {0.17, 0.2};
    private final double   scalex       = 6.0;
    private final double   scaley       = 3.0;
    private final double   scalez       = 4.0;
    private final double[] amplitudesx  = new double[frequenciesx.length];
    private final double[] amplitudesy  = new double[frequenciesy.length];
    private final double[] amplitudesz  = new double[frequenciesz.length];
    private final double[] phasesx      = new double[frequenciesx.length];
    private final double[] phasesy      = new double[frequenciesy.length];
    private final double[] phasesz      = new double[frequenciesz.length];
    
    public Vector3d getOrigin() {
        return this.origin;
    }
    
    public Vector3d getPosition() {
        return this.position;
    }
    
    public Vector3d getRelativePosition() {
        return this.relposition;
    }
    
    public boolean isDestroyed() {
        return this.destroyed;
    }
    
    public void reset() {
        this.t = 0L;
        this.destroyed = false;
        //
        final Random rnd = new Random(System.currentTimeMillis());
        //
        // randomize phases.
        //
        for (int i = 0; i < this.phasesx.length; i++) {
            this.phasesx[i] = Math.PI * rnd.nextDouble() * 2.0;
        }
        for (int i = 0; i < this.phasesy.length; i++) {
            this.phasesy[i] = Math.PI * rnd.nextDouble() * 2.0;
        }
        for (int i = 0; i < this.phasesz.length; i++) {
            this.phasesz[i] = Math.PI * rnd.nextDouble() * 2.0;
        }
        //
        this.updatePosition();
    }
    
    public EnemySpaceShip() {
        //
        for (int i = 0; i < this.amplitudesx.length; i++) {
            this.frequenciesx[i] *= 0.2;
        }
        for (int i = 0; i < this.amplitudesy.length; i++) {
            this.frequenciesy[i] *= 0.2;
        }
        for (int i = 0; i < this.amplitudesz.length; i++) {
            this.frequenciesz[i] *= 0.2;
        }
        //
        for (int i = 0; i < this.amplitudesx.length; i++) {
            this.amplitudesx[i] = this.scalex;
        }
        for (int i = 0; i < this.amplitudesy.length; i++) {
            this.amplitudesy[i] = this.scaley;
        }
        for (int i = 0; i < this.amplitudesz.length; i++) {
            this.amplitudesz[i] = this.scalez;
        }
        for (int i = 0; i < this.phasesy.length; i++) {
            this.phasesy[i] = Math.PI;
        }
        for (int i = 0; i < this.phasesz.length; i++) {
            this.phasesz[i] = 0.5 * Math.PI;
        }
        //
        this.reset();
    }
    
    private static double f(
        final double t,
        final double[] frequencies,
        final double[] amplitudes,
        final double[] phase
    ) {
        double sum = 0.0;
        //
        for (int i = 0; i < frequencies.length; i++) {
            sum += (amplitudes[i] * Math.sin((t * frequencies[i]) + phase[i]));
        }
        //
        return sum;
    }
    
    public void destroy() {
        this.destroyed = true;
    }
    
    public void update() {
        
        this.updatePosition();
        if (!this.destroyed) this.t += 1;
    }
    
    private void updatePosition() {
        //
        /* */
        this.relposition.x = f(
            this.t,
            this.frequenciesx,
            this.amplitudesx,
            this.phasesx
        );
        this.relposition.y = f(
            this.t,
            this.frequenciesy,
            this.amplitudesy,
            this.phasesy
        );
        this.relposition.z = f(
            this.t,
            this.frequenciesz,
            this.amplitudesz,
            this.phasesz
        );
        /* */
        //
        this.position.x = this.origin.x + this.relposition.x;
        this.position.y = this.origin.y + this.relposition.y;
        this.position.z = this.origin.z + this.relposition.z;
    }
    

    
}