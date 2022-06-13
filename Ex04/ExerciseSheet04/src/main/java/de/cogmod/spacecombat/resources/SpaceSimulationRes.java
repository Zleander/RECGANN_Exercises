package de.cogmod.spacecombat.resources;


import java.awt.image.BufferedImage;
import java.net.URISyntaxException;
import java.net.URL;

import javax.imageio.ImageIO;

/**
 * @author Sebastian Otte
 */
public class SpaceSimulationRes {
	
	public static final BufferedImage IMG_ENEMY;
	public static final double        IMG_ENEMY_REL;
	
	public static final BufferedImage IMG_SPACE;
	
	public static final BufferedImage IMG_COCKPIT;

	public static final BufferedImage IMG_MISSILE;

	public static final BufferedImage IMG_EXPLOSION;
	public static final double        IMG_EXPLOSION_REL;
	public static final int[][]       IMG_EXPLOSION_FRAMES;
    
	public static final URL load(final String resource) throws URISyntaxException {
    	final URL url = (
    		ClassLoader.getSystemClassLoader().getResource(resource)
    	);
    	return url;
    }
    
	static {
		try {
			//
			IMG_ENEMY     = ImageIO.read(load("de/cogmod/spacecombat/resources/tiefighter.png"));
			IMG_ENEMY_REL = ((double)(IMG_ENEMY.getHeight()) / ((double)(IMG_ENEMY.getWidth())));
			//
			IMG_SPACE     = ImageIO.read(load("de/cogmod/spacecombat/resources/space.jpg"));
            //
            IMG_COCKPIT   = ImageIO.read(load("de/cogmod/spacecombat/resources/cockpit.png"));
            //
            IMG_MISSILE   = ImageIO.read(load("de/cogmod/spacecombat/resources/missile.png"));
            //
            IMG_EXPLOSION = ImageIO.read(load("de/cogmod/spacecombat/resources/explosion.png"));
            //
            {
                final int framesx     = 8;
                final int framesy     = 6;
                final int frames      = framesx * framesy;
                final int framewidth  = IMG_EXPLOSION.getWidth() / framesx;
                final int frameheight = IMG_EXPLOSION.getHeight() / framesy;
                //
                IMG_EXPLOSION_FRAMES = new int[frames][];
                //
                int frame = 0;
                //
                for (int i = 0; i < framesy; i++) {
                    for (int j = 0; j < framesx; j++) {
                        IMG_EXPLOSION_FRAMES[frame++] = new int[]{
                            j * framewidth,
                            i * frameheight,
                            framewidth,
                            frameheight
                        };
                    }
                }
                IMG_EXPLOSION_REL = ((double)(frameheight) / ((double)(framewidth)));
            }
            
            //
		} catch (final Exception e) {
			throw new RuntimeException(e);
		}
	}
	
}