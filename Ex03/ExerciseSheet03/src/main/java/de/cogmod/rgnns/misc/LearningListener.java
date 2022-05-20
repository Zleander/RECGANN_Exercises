package de.cogmod.rgnns.misc;

/**
 * @author Sebastian Otte
 */
public interface LearningListener {
    public void afterEpoch(final int epoch, final double trainingerror);
}