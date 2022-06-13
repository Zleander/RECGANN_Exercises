package de.cogmod.rgnns.math;

/**
 * This class is a vector of three double values x, y and z.
 * It contains some static methods for operating in R3 (3 dimensional space).
 * <br></br>
 * @author Sebastian Otte
 */
public class Vector3d {

	public double x = 0.0f;
    public double y = 0.0f;
    public double z = 0.0f;

    @Override
    public boolean equals(Object pvec) {
    	if (!(pvec instanceof Vector3d)) return false;     		

    	Vector3d vec = (Vector3d)pvec;
    	return ((this.x == vec.x) && (this.y == vec.y) && (this.z == vec.z)); 
    }
    /**
     * Copies the values of a given Vector3d.
     * @param pv A vector for copy.
     */
    public void set(Vector3d pv) {
    	this.x = pv.x;
    	this.y = pv.y;
    	this.z = pv.z;
    }
    
    public Vector3d copy() {
        return new Vector3d(this.x, this.y, this.z);
    }
    
    /**
     * Constructs Vector3d by three double parameter. 
     * @param px The x value.
     * @param py The y value.
     * @param pz The z value.
     */
    public Vector3d(double px, double py, double pz)  {
    	this.x = px;
    	this.y = py;
    	this.z = pz;
    }
    /**
     * Constructs Vector3d by an other Vector3d. 
     * @param pv Reference of other Vector3d.
     */
    public Vector3d(Vector3d pv) {
    	this.x = pv.x;
    	this.y = pv.y;
    	this.z = pv.z;
    }
    /**
     * Constructs an 0 initialized Vector3d. 
     */
    public Vector3d() {
        //
    }
    /**
     * Return the length of the vector in R3.
     * @return Vector length.
     */
    public double length() {
        return (double)Math.sqrt((this.x*this.x) + 
        						(this.y*this.y) + 
        						(this.z*this.z));
    }
    /**
     * Return the length of the vector in R3 without sqrt.
     * @return Vector length.
     */
    public double length2() {
        return (this.x * this.x) + 
        	   (this.y * this.y) + 
        	   (this.z * this.z);
    }
    
	// ----------------------------------------------------------------
	// static methods
	// ----------------------------------------------------------------
    
    /**
     * Adds two vectors and stores the result into third vector.
     * @param pv1 The left operand vector. 
     * @param pv2 The right operand vector.
     * @param pvret The result vector.
     */
    public static void add(Vector3d pv1, Vector3d pv2, Vector3d pvret) {
        pvret.x = pv1.x + pv2.x;
        pvret.y = pv1.y + pv2.y;
        pvret.z = pv1.z + pv2.z;
    }
    /**
     * Adds two vectors and returns a new vector within the result.
     * @param pv1 The left operand vector. 
     * @param pv2 The right operand vector.
     * @return A new vector within the result of the operation.
     */
    public static Vector3d add(Vector3d pv1, Vector3d pv2) {
        return new Vector3d(pv1.x + pv2.x, pv1.y + pv2.y, pv1.z + pv2.z);
    }
    /**
     * Subtracts two vectors and stores the result into third vector.
     * @param pv1 The left operand vector. 
     * @param pv2 The right operand vector.
     * @param pvret The result vector.
     */
    public static void sub(Vector3d pv1, Vector3d pv2, Vector3d pvret) {
        pvret.x = pv1.x - pv2.x;
        pvret.y = pv1.y - pv2.y;
        pvret.z = pv1.z - pv2.z;
    }
    /**
     * Subtracts two vectors and returns a new vector within the result.
     * @param pv1 The left operand vector. 
     * @param pv2 The right operand vector.
     * @return A new vector within the result of the operation.
     */
    public static Vector3d sub(Vector3d pv1, Vector3d pv2) {
        return new Vector3d(pv1.x - pv2.x, pv1.y - pv2.y, pv1.z - pv2.z);
    }
    /**
     * Multiplies a vector with a scalar and stores the result in second vector.
     * @param pv The left operand vector. 
     * @param psc The right operand scalar.
     * @param pvret The result vector.
     */
    public static void mul(Vector3d pv, double psc, Vector3d pvret) {
        pvret.x = pv.x * psc;
        pvret.y = pv.y * psc;
        pvret.z = pv.z * psc;
    }
    /**
     * Multiplies a vector with a scalar and returns the result as new vector.
     * @param pv The left operand vector. 
     * @param psc The right operand scalar.
     * @return A new vector within the result of the operation.
     */
    public static Vector3d mul(Vector3d pv, double psc) {
        return new Vector3d(pv.x * psc, pv.y * psc, pv.z * psc);
    }
    /**
     * Divides a vector by a scalar and stores the result in second vector.
     * @param pv The left operand vector. 
     * @param psc The right operand scalar.
     * @param pvret The result vector.
     */
    public static void div(Vector3d pv, double psc, Vector3d pvret) {
        double iv = 1 / psc;
        pvret.x = pv.x * iv;
        pvret.y = pv.y * iv;
        pvret.z = pv.z * iv;
    }
    /**
     * Divides a vector by a scalar and returns the result as new vector.
     * @param pv The left operand vector. 
     * @param psc The right operand scalar.
     * @return A new vector within the result of the operation.
     */    
    public static Vector3d div(Vector3d pv, double psc)  {
        double iv = 1 / psc;
        return new Vector3d(pv.x * iv, pv.y * iv, pv.z * iv);
    }
    /**
     * Normalizes a vector and stores the result that is a vector
     * in the same direction with length of 1 in pvret.
     * @param pv Source vector.
     * @param pvret Normalized vector.
     */
    public static void normalize(Vector3d pv, Vector3d pvret) {
        double l = 1 / pv.length();
        pvret.x = pv.x * l;
        pvret.y = pv.y * l;
        pvret.z = pv.z * l; 
    }
    /**
     * Normalizes a vector and returns the result that is a vector
     * in the same direction with length of 1 as a new vector.
     * @param pv The source vector.
     * @return A new normalized vector.
     */
    public static Vector3d normalize(Vector3d pv) {
        final double length = pv.length();
        final double il = (length > 0.0)?(1.0 / length):(0.0);
        return new Vector3d(pv.x * il, pv.y * il, pv.z * il);
    }
    /**
     * Inverts a given vector and stores the result into pvret.
     * @param pv The source vector.
     * @param pvret Inverted vector.
     */
    public static void invert(Vector3d pv, Vector3d pvret) {
        pvret.x = -pv.x;
        pvret.y = -pv.y;
        pvret.z = -pv.z;
    }
    /**
     * Inverts a given vector and returns a new inverted vector.
     * @param pv The source vector.
     * @return A new inverted vector.
     */
    public static Vector3d invert(Vector3d pv) {
        return new Vector3d(-pv.x, -pv.y, -pv.z);
    }
    /**
     * Returns the scalar product of two vectors.
     * @param pv1 Left operand vector.
     * @param pv2 Right operand vector.
     * @return The scalar product.
     */
    public static double scalar(Vector3d pv1, Vector3d pv2) {
        return (pv1.x * pv2.x) + (pv1.y * pv2.y) + (pv1.z * pv2.z);
    }
    /**
     * Builds the cross product of two vectors an stores the result
     * into pvret.
     * @param pv1 Left operand vector.
     * @param pv2 Right operand vector.
	 * @param pvret The result vector.
     */
    public static void cross(Vector3d pv1, Vector3d pv2, Vector3d pvret) {
        pvret.x = (pv1.y * pv2.z) - (pv1.z * pv2.y);
        pvret.y = (pv1.z * pv2.x) - (pv1.x * pv2.z);
        pvret.z = (pv1.x * pv2.y) - (pv1.y * pv2.x);
    }
    /**
     * Returns the cross product of two vectors. 
     * @param pv1 Left operand vector.
     * @param pv2 Right operand vector.
     * @return A new vector within the result of the operation.
     */
    public static Vector3d cross(Vector3d pv1, Vector3d pv2) {
        return new Vector3d((pv1.y * pv2.z) - (pv1.z * pv2.y),
                            (pv1.z * pv2.x) - (pv1.x * pv2.z),
                            (pv1.x * pv2.y) - (pv1.y * pv2.x));
    }
    /**
     * Returns the angle (phi) between two vectors.
     * @param pv1 The first vector.
     * @param pv2 The second vector.
     * @return The angle.
     */
    public static double phi(Vector3d pv1, Vector3d pv2) {
        return (double)Math.acos(scalar(pv1, pv2) / 
                                (pv1.length() * pv2.length()));
    }
    /**
     * Rotates a vector around the x-axis and stores the result into pvret.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @param pvret The return vector.
     */    
    public static void rotateX(Vector3d pv, double pangle, Vector3d pvret) {
        double cosa = (double)Math.cos(pangle);
        double sina = (double)Math.sin(pangle);
        
        pvret.x = pv.x;
        pvret.y = (pv.y * cosa) + (pv.z * -sina);  
        pvret.z = (pv.y * sina) + (pv.z * cosa);         
    }
    /**
     * Rotates a vector around the x-axis and returns the result as new vector.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @return The return vector.
     */    
    public static Vector3d rotateX(Vector3d pv, double pangle) {
        Vector3d v = new Vector3d();
        rotateX(pv, pangle, v);
        return v;
    }
    /**
     * Rotates a vector around the y-axis and stores the result into pvret.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @param pvret The return vector.
     */    
    public static void rotateY(Vector3d pv, double pangle, Vector3d pvret) {
        double cosa = (double)Math.cos(pangle);
        double sina = (double)Math.sin(pangle);
        
        pvret.x = (pv.x * cosa) + (pv.z * sina);
        pvret.y = pv.y;  
        pvret.z = (pv.x * -sina) + (pv.z * cosa);         
    }
    /**
     * Rotates a vector around the y-axis and returns the result as new vector.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @return The return vector.
     */    
    public static Vector3d rotateY(Vector3d pv, double pangle) {
        Vector3d v = new Vector3d();
        rotateY(pv, pangle, v);
        return v;        
    }
    /**
     * Rotates a vector around the z-axis and stores the result into pvret.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @param pvret The return vector.
     */    
    public static void rotateZ(Vector3d pv, double pangle, Vector3d pvret) {
        //
    	double cosa = (double)Math.cos(pangle);
        double sina = (double)Math.sin(pangle);
        
        pvret.x = (pv.x * cosa) + (pv.y * -sina);
        pvret.y = (pv.x * sina) + (pv.y * cosa);  
        pvret.z = pv.z;  
    }
    /**
     * Rotates a vector around the z-axis and returns the result as new vector.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @return The return vector.
     */    
    public static Vector3d rotateZ(Vector3d pv, double pangle) {
        //
    	Vector3d v = new Vector3d();
        rotateZ(pv, pangle, v);
        return v;        
    }
    
    /**
     * The methods builds the normalized plane normal of the input vectors and stores
     * the result into pvret. Note that this method uses the right hand rule.
     * <br></br>
     * @param pvec1 The first vector "a".
     * @param pvec2 The second vector "b".
     * @param pvec3 The third vector "c".
     * @param pret
     */
    public static void normal(Vector3d pvec1, Vector3d pvec2, Vector3d pvec3, Vector3d pvret) {
    	//
    	// Build plane with two vectors with source point pvec1.
    	//
    	Vector3d ab = Vector3d.sub(pvec2, pvec1);
    	Vector3d ac = Vector3d.sub(pvec3, pvec1);
    	//
    	// Build the cross product to get the plane normal, and normalize the vector.
    	//
    	Vector3d.cross(ab, ac, pvret);
    	Vector3d.normalize(pvret, pvret);
    }
    
    /**
     * The methods builds the normalized plane normal of the input vectors and returns
     * the result as a new vector. Note that this method uses the right hand rule.
     * <br></br>
     * @param pvec1 The first vector "a".
     * @param pvec2 The second vector "b".
     * @param pvec3 The third vector "c".
     * @param pret
     */
    public static Vector3d normal(Vector3d pvec1, Vector3d pvec2, Vector3d pvec3) {
    	//
    	Vector3d ret = new Vector3d();
    	normal(pvec1, pvec2, pvec3, ret);
    	return ret;
    }
    
    
}