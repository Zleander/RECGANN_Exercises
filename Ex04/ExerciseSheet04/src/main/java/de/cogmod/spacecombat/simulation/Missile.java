package de.cogmod.spacecombat.simulation;


import de.cogmod.rgnns.math.Vector3d;
import de.jannlab.math.MathTools;

/**
 * @author Sebastian Otte
 */
public class Missile {
    
    public static double ACC_INITIAL = 0.01;
    public static double ACC_DECAY   = 0.99;
    
    public static long MAX_LIFETIME = 200;
    
    private Vector3d position    = new Vector3d();
    private Vector3d direction   = new Vector3d();
    private double   acc         = ACC_INITIAL;
    private Vector3d velocity    = new Vector3d();
    private boolean  launched    = false;
    private boolean  destroyed   = false;
    private long     timetolive  = MAX_LIFETIME; 
    
    private double adjustx = 0.0;
    private double adjusty = 0.0;
    
    public boolean isLaunched() {
        return this.launched;
    }
    
    public Vector3d getPosition() {
        return this.position;
    }
    
    public long getTimeToLive() {
        return this.timetolive;
    }
    
    public void destroy() {
        this.destroyed = true;
    }
    
    public void reset() {
        this.position.x = 0.0;
        this.position.y = 0.0;
        this.position.z = 0.0;
        //
        this.direction.x = 0.0;
        this.direction.y = 0.0;
        this.direction.z = 1.0;
        //
        this.velocity.x = 0.0;
        this.velocity.y = 0.0;
        this.velocity.z = 0.0;
        //
        this.adjustx = 0.0;
        this.adjusty = 0.0;
        //
        this.launched   = false;
        this.destroyed  = false;
        this.acc        = ACC_INITIAL;
        this.timetolive = MAX_LIFETIME;
    }
    
    public double getAdjustX() {
        return this.adjustx;
    }
    
    public double getAdjustY() {
        return this.adjusty;
    }
    
    public boolean isDestroyed() {
        return this.destroyed;
    }
    
    public void launch() {
        if (!this.destroyed) this.launched = true;
    }
    
    public Vector3d getDirection() {
        return this.direction;
    }
    
    public Missile() {
        this.reset();
    }
    
    public void adjust(final double x, final double y) {
        this.adjustx = MathTools.clamp(x, -1, 1);
        this.adjusty = MathTools.clamp(y, -1, 1);
    }
    
    private double computeVel(
        final double v,
        final double vl,
        final double acc,
        final double u
    ) {
        return (vl + acc) * u;
    }
    
    public static double sq(final double x) {
        return x * x;
    }
    
    public void update() {
        if (!this.destroyed && this.launched) {
            //
            final double vl = this.velocity.length();
            final double fade = 0.05;
            //
            // adapt direction vectors.
            //
            final double vax = this.adjustx * fade;
            final double vay = this.adjusty * fade;
            //
            this.adjustx = 0.0;
            this.adjusty = 0.0;
            //
            Vector3d.rotateX(this.direction.copy(), vax, this.direction); 
            Vector3d.rotateY(this.direction.copy(), vay, this.direction);
            Vector3d.normalize(this.direction.copy(), this.direction);
            //
            this.velocity.x = computeVel(this.velocity.x, vl, this.acc, this.direction.x);
            this.velocity.y = computeVel(this.velocity.y, vl, this.acc, this.direction.y);
            this.velocity.z = computeVel(this.velocity.z, vl, this.acc, this.direction.z);
            //
            this.position.x += this.velocity.x;
            this.position.y += this.velocity.y;
            this.position.z += this.velocity.z;
            //
            this.acc *= ACC_DECAY;
            //
            if (this.timetolive <= 0) {
                this.timetolive = 0;
                this.launched  = false;
                this.destroyed = true;
            } else {
                this.timetolive--;
            }
        }
    }
    
}