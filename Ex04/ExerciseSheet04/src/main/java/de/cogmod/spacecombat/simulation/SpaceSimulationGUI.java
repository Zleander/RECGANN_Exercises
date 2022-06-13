package de.cogmod.spacecombat.simulation;


import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

import de.cogmod.rgnns.math.Vector3d;
import de.cogmod.spacecombat.AIMComputer;
import de.cogmod.spacecombat.resources.SpaceSimulationRes;

/**
 * @author Sebastian Otte
 */
public class SpaceSimulationGUI extends JFrame {
    private static final long serialVersionUID = 5247220512930428467L;

    public static String TITLE = "RNN Wars - The ESN Strikes Back";
        
    private BufferedImage buffer;
    private Graphics2D g;
    
    private SpaceSimulation sim;
    private AIMComputer aimcomputer;
    
    private Missile missile   = null;
    //
    private boolean keyup    = false;
    private boolean keyleft  = false;
    private boolean keyright = false;
    private boolean keydown  = false;
    private boolean pause    = false;

    private Timer timer;
    
    private final Color  aimcolor  = new Color(255, 0, 0, 120);
    private final Stroke aimstroke = new BasicStroke(
        4.0f
    );
    private final Color  predictioncolor  = new Color(0, 255, 0, 120);
    private final Stroke predictionstroke = new BasicStroke(
        3.0f
    );
    
    private final int    explosionkickin  = 6;
    private final int    explosionfadeout = 100;
    private final double explosionscale   = 20.0f;
    
    private void drawBackground() {
        //
        // draw background....
        //
        this.g.drawImage(
            SpaceSimulationRes.IMG_SPACE,
            0,
            0,
            SpaceSimulationRes.IMG_SPACE.getWidth(),
            SpaceSimulationRes.IMG_SPACE.getHeight(),
            null
        );
    }
   
    private void drawMissile(final Missile r) {
        //
        if (r.getPosition().z < 0) return;
        //
        final int centerx = this.buffer.getWidth() / 2;
        final int centery = this.buffer.getHeight() / 2;
        //
        final double missilezscale = 1.0 / r.getPosition().z;
        //
        final double projmissileposx = r.getPosition().x * missilezscale;
        final double projmissileposy = r.getPosition().y * missilezscale;
        //
        final int missileposx = centerx + (int)(projmissileposx * centerx);
        final int missileposy = centery - (int)(projmissileposy * centery);
        //
        this.g.setColor(Color.BLUE);
        //
        //
        final int missilewidth  = (int)(this.buffer.getWidth() * missilezscale);
        final int missileheight = missilewidth;
        //
        this.g.drawImage(
            SpaceSimulationRes.IMG_MISSILE,
            missileposx - (missilewidth / 2),
            missileposy - (missileheight / 2),
            missileposx + (missilewidth / 2),
            missileposy + (missileheight / 2),
            0,
            0,
            SpaceSimulationRes.IMG_MISSILE.getWidth(),
            SpaceSimulationRes.IMG_MISSILE.getHeight(),
            null
        );
    }
    

    private void drawEnemy() {
        //
        if (this.sim.getEnemy().isDestroyed()) return;
        //
        final int centerx   = this.buffer.getWidth() / 2;
        //final int hwidth    = centerx;
        final int centery   = this.buffer.getHeight() / 2;
        //final int hheight   = centery;
        //
        final EnemySpaceShip enemy = this.sim.getEnemy();
        
        final double enemyzscale = 1.0 / enemy.getPosition().z;
        //
        final double projenemyposx = enemy.getPosition().x * enemyzscale;
        final double projenemyposy = enemy.getPosition().y * enemyzscale;
        //
        final int enemyposx = centerx + (int)(projenemyposx * centerx);
        final int enemyposy = centery - (int)(projenemyposy * centery);
        //
        this.g.setColor(Color.BLUE);
        //
        final int enemywidth  = (int)(this.buffer.getWidth() * enemyzscale);
        final int enemyheight = (int)(SpaceSimulationRes.IMG_ENEMY_REL * enemywidth);
        //
        this.g.drawImage(
            SpaceSimulationRes.IMG_ENEMY,
            enemyposx - (enemywidth / 2),
            enemyposy - (enemyheight / 2),
            enemyposx + (enemywidth / 2),
            enemyposy + (enemyheight / 2),
            0,
            0,
            SpaceSimulationRes.IMG_ENEMY.getWidth(),
            SpaceSimulationRes.IMG_ENEMY.getHeight(),
            null
        );
    }
 
    private void drawMissilesFront() {
        for (Missile r : this.sim.getMissiles()) {
            if (
                !r.isDestroyed() && 
                r.isLaunched() &&
                r.getPosition().z <= this.sim.getEnemy().getPosition().z
            ) {
                this.drawMissile(r);
            }
        }        
    }
    
    private void drawMissilesBack() {
        for (Missile r : this.sim.getMissiles()) {
            if (
                !r.isDestroyed() && 
                r.isLaunched() &&
                r.getPosition().z > this.sim.getEnemy().getPosition().z
            ) {
                this.drawMissile(r);
            }
        }     
    }
    
    private void drawCockpit() {
        this.g.drawImage(
            SpaceSimulationRes.IMG_COCKPIT,
            0,
            0,
            SpaceSimulationRes.IMG_COCKPIT.getWidth(),
            SpaceSimulationRes.IMG_COCKPIT.getHeight(),
            null
        );          
    }
    
    private static double safeDiv(final double x, final double d) {
        return (d == 0.0)?(1.0):(x / d);
    }
    
    private void drawTrajectory(final Vector3d start, final Vector3d[] traj) {
        //
        final int centerx = this.buffer.getWidth() / 2;
        final int centery = this.buffer.getHeight() / 2;
        //
        double lastzscale   = safeDiv(1.0, start.z);
        double lastprojposx = start.x * lastzscale;
        double lastprojposy = start.y * lastzscale;
        //
        int lastposx = centerx + (int)(lastprojposx * centerx);
        int lastposy = centery - (int)(lastprojposy * centery);
        //
        for (int i = 0; i < traj.length; i++) {
            final Vector3d curr = traj[i];
            final double zscale = safeDiv(1.0, curr.z);
            //
            final double projposx = curr.x * zscale;
            final double projposy = curr.y * zscale;
            //
            final int posx = centerx + (int)(projposx * centerx);
            final int posy = centery - (int)(projposy * centery);
            //
            this.g.drawLine(lastposx, lastposy, posx, posy);
            //
            lastposx = posx;
            lastposy = posy;
        }            
    }
    
    private void drawHUD() {
        //
        final int centerx = this.buffer.getWidth() / 2;
        //final int hwidth    = centerx;
        final int centery = this.buffer.getHeight() / 2;
        //
        // draw enemy target prediction lock/aim visualization
        //
        if (!this.sim.getEnemy().isDestroyed() && 
            this.aimcomputer.getTargetLocked()
        ) {
            //
            final EnemySpaceShip enemy = this.aimcomputer.getTarget();
            //
            if (this.aimcomputer.getEnemyTrajectoryPrediction() != null) {
                final Vector3d[] pred = this.aimcomputer.getEnemyTrajectoryPrediction();
                this.g.setStroke(this.predictionstroke);
                this.g.setColor(this.predictioncolor);
                drawTrajectory(enemy.getPosition(), pred);
            }
            //
            double enemyzscale   = safeDiv(1.0, enemy.getPosition().z);
            double enemyprojposx = enemy.getPosition().x * enemyzscale;
            double enemyprojposy = enemy.getPosition().y * enemyzscale;
            //
            int enemyposx = centerx + (int)(enemyprojposx * centerx);
            int enemyposy = centery - (int)(enemyprojposy * centery);
            //
            this.g.setStroke(this.aimstroke);
            this.g.setColor(this.aimcolor);
            //
            this.g.drawOval(enemyposx - 35, enemyposy - 35, 70, 70);
            final int lmin = 22;
            final int lmax = 28;
            this.g.drawLine(
                enemyposx - lmax, enemyposy - lmax, 
                enemyposx - lmin, enemyposy - lmin
            );
            this.g.drawLine(
                enemyposx + lmax, enemyposy - lmax, 
                enemyposx + lmin, enemyposy - lmin
            );
            this.g.drawLine(
                enemyposx + lmax, enemyposy + lmax, 
                enemyposx + lmin, enemyposy + lmin
            );
            this.g.drawLine(
                enemyposx - lmax, enemyposy + lmax, 
                enemyposx - lmin, enemyposy + lmin
            );
        }
        
    }
    
    private void drawExplosions() {
        for (Explosion explosion : this.sim.getExplosions()) {
            //
            final int age = (int)(
                this.sim.getGlobalTime() - explosion.getStartTime()
            );
            final int frame = (age / 3);
            if (
                (frame < 0) || 
                (frame >= SpaceSimulationRes.IMG_EXPLOSION_FRAMES.length) 
            ) {
                continue; 
            }
            //
            final int centerx   = this.buffer.getWidth() / 2;
            final int centery   = this.buffer.getHeight() / 2;
            //
            final Vector3d position = explosion.getPosition();
            
            final double zscale = 1.0 / position.z;
            //
            final double projposx = position.x * zscale;
            final double projposy = position.y * zscale;
            //
            final int posx = centerx + (int)(projposx * centerx);
            final int posy = centery - (int)(projposy * centery);
            //
            this.g.setColor(Color.BLUE);
            //
            final int width = (int)(
                this.buffer.getWidth() * zscale * this.explosionscale
             );
            final int height = (int)(SpaceSimulationRes.IMG_EXPLOSION_REL * width);
            //
            final int[] frameroi = SpaceSimulationRes.IMG_EXPLOSION_FRAMES[frame];
            //
            this.g.drawImage(
                SpaceSimulationRes.IMG_EXPLOSION,
                posx - (width / 2),
                posy - (height / 2),
                posx + (width / 2),
                posy + (height / 2),
                frameroi[0],
                frameroi[1],
                frameroi[0] + frameroi[2] - 1,
                frameroi[1] + frameroi[3] - 1,
                null
            );
            //
        }
    }
    
    private void drawInCockpitExplosionEffects() {
        //
        for (Explosion explosion : this.sim.getExplosions()) {
            final long age = (
                this.sim.getGlobalTime() - explosion.getStartTime() - 
                this.explosionkickin
            );
            float fade = 0;
            if (age >= 0) {
                fade = Math.max(
                    0.0f, 
                    1.0f - (
                        (float)(age) / (float)(this.explosionfadeout)
                    )
                );
            } else {
                fade = 1.0f - (-(float)(age) / this.explosionkickin);
            }
            final Color dazzlecolor = new Color(
                1.0f, 1.0f, 1.0f, 0.9f * fade
            );
            this.g.setColor(dazzlecolor);
            this.g.fillRect(0, 0, this.getWidth(), this.getHeight());
        }
            
    }
    
    public void renderScene() {
        this.drawBackground();
        this.drawMissilesBack();
        this.drawEnemy();
        this.drawExplosions();
        this.drawMissilesFront();
        this.drawHUD();
        this.drawCockpit();
        this.drawInCockpitExplosionEffects();
    }
    
    
    public SpaceSimulationGUI(
        final SpaceSimulation sim,
        final AIMComputer aimcomputer
    ) {
        super(TITLE);
        //
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.sim         = sim;
        this.aimcomputer = aimcomputer;
        //
        // create painting buffer.
        //
        this.buffer = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        //
        // draw panel.
        //
        final JPanel panel = new JPanel() {
            private static final long serialVersionUID = -4307908552010057652L;
            @Override
            protected void paintComponent(final Graphics gfx) {
                super.paintComponent(gfx);
                gfx.drawImage(
                    buffer,  
                    0,  0, 
                    buffer.getWidth(), buffer.getHeight(),  null
                );
            }
        };
        //
        this.g = (Graphics2D)buffer.getGraphics();
        g.setRenderingHint(
            RenderingHints.KEY_ANTIALIASING,
            RenderingHints.VALUE_ANTIALIAS_ON
        );
        //
        panel.setPreferredSize(new Dimension(buffer.getWidth(), buffer.getHeight()));
        this.add(panel);
        this.setResizable(false);
        this.pack();
        //
        this.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                //
                super.keyPressed(e);
                //
                switch (e.getKeyCode()) {
                    case KeyEvent.VK_UP:
                        keyup = true;
                        break;
                    case KeyEvent.VK_LEFT:
                        keyleft= true;
                        break;
                    case KeyEvent.VK_RIGHT:
                        keyright = true;
                        break;
                    case KeyEvent.VK_DOWN:
                        keydown = true;
                        break;
                    case KeyEvent.VK_ENTER:
                        missile = sim.launchMissile();
                        break;
                    case KeyEvent.VK_L:
                        if (!aimcomputer.getTargetLocked()) {
                            aimcomputer.lockTarget(sim.getEnemy());
                            System.out.println("target locked.");
                        } else {
                            aimcomputer.releaseTarget();
                            System.out.println("target released.");
                        }
                        break;
                    case KeyEvent.VK_P:
                        pause = !pause;
                        break;
                    case KeyEvent.VK_R:
                        sim.reset();
                        break;
                }
            }
            @Override
            public void keyReleased(KeyEvent e) {
                switch (e.getKeyCode()) {
                    case KeyEvent.VK_UP:
                        keyup    = false;
                        break;
                    case KeyEvent.VK_LEFT:
                        keyleft  = false;
                        break;
                    case KeyEvent.VK_RIGHT:
                        keyright = false;
                        break;
                    case KeyEvent.VK_DOWN:
                        keydown  = false;
                        break;
                }
            }            
            
        });
        //
        final double fps    = 30.0;
        final double dtmsec = 1000.0 / fps;
        //
        this.timer = new Timer((int)(dtmsec), new ActionListener() {
            //
            @Override
            public void actionPerformed(final ActionEvent e) {
                //
                if (
                    missile!= null && 
                    missile.isLaunched() && 
                    !missile.isDestroyed()
                ) {
                    double rotx = missile.getAdjustX();
                    double roty = missile.getAdjustY();
                    //
                    if (keyup)    rotx -= 1.0;
                    if (keydown)  rotx += 1.0;
                    if (keyleft)  roty -= 1.0;
                    if (keyright) roty += 1.0;
                    //
                    missile.adjust(rotx, roty);
                }
                //
                if (!pause) {
                    sim.update();
                    //
                    SpaceSimulationGUI.this.renderScene();
                    //
                    panel.repaint();
                }
            }
        });
    }
    
    public void start() {
        this.timer.start();
    }
    
}