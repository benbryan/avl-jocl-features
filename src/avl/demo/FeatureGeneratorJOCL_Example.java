package avl.demo;

import avl.sv.shared.model.featureGenerator.jocl.feature.Feature;
import avl.sv.shared.model.featureGenerator.jocl.FeatureGeneratorJOCL;
import avl.sv.shared.model.featureGenerator.jocl.feature.GaborKernal;
import avl.sv.shared.model.featureGenerator.jocl.plane.Plane;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.GroupLayout;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

public class FeatureGeneratorJOCL_Example extends javax.swing.JFrame {
   
    private void kernelPlotter() {
                
        GaborKernal kernel = FeatureGeneratorJOCL.gaborKernal(Feature.Names.Gabor_d0r0);
        
        String featureNames[] = new String[]{"real", "imag"};
        float results[][] = new float[2][];
        results[0] = kernel.getReal();
        results[1] = kernel.getImag();
                
        int numelKernels = results.length;
        float ll[] = new float[numelKernels];
        float ul[] = new float[numelKernels];
        
        float min[] = new float[numelKernels];
        float max[] = new float[numelKernels];
        for (int i = 0; i < results.length; i++){
            min[i] = Float.POSITIVE_INFINITY;
            max[i] = Float.NEGATIVE_INFINITY;            
        }
        for (int i = 0; i < results.length; i++){
            for (int v = 0; v < results[i].length; v++){
                if (results[i][v]<min[i]){
                    min[i]=results[i][v];
                }
                if (results[i][v]>max[i]){
                    max[i]=results[i][v];
                }
            }
        }
        for (int i = 0; i < results.length; i++){
            ll[i] = min[i];
            ul[i] = max[i];
            System.out.println("min=" + String.valueOf(ll[i]));
            System.out.println("max=" + String.valueOf(ul[i]));
        }
        
        int boxSize = 6;
        final BufferedImage resultImgs[] = new BufferedImage[numelKernels];
        Graphics graphics[] = new Graphics[numelKernels];
        for (int f = 0; f < resultImgs.length; f++) {
            resultImgs[f] = new BufferedImage(kernel.getKernalSize()*boxSize, kernel.getKernalSize()*boxSize, BufferedImage.TYPE_BYTE_GRAY);
            graphics[f] = resultImgs[f].getGraphics();
        }

        for (int i = 0; i < resultImgs.length; i++) {
            addTab(resultImgs[i], featureNames[i]);
        }
        
        for (int kernelIdx = 0; kernelIdx < results.length; kernelIdx++) {
            for (int x = 0; x < kernel.getKernalSize(); x++){
                for (int y = 0; y < kernel.getKernalSize(); y++){   
                    float fVal = results[kernelIdx][x+kernel.getKernalSize()*y];
                    fVal = (fVal - ll[kernelIdx]) / (ul[kernelIdx] - ll[kernelIdx]);
                    fVal = Math.min(1, fVal);
                    fVal = Math.max(0, fVal);
                    float fValFloat = (float) fVal;
//                    System.out.println(String.valueOf(fVal));
                    graphics[kernelIdx].setColor(new Color(fValFloat, fValFloat, fValFloat));
                    graphics[kernelIdx].fillRect(x*boxSize, y*boxSize, boxSize, boxSize);
                }
            }
        }        
    }    
    
    class Sample {
        final int idx, x, y;
        final BufferedImage img;
        public Sample(int idx, int x, int y, BufferedImage img) {
            this.idx = idx;
            this.x = x;
            this.y = y;
            this.img = img;
        }
    }

    private ArrayList<Plane> getAllPlanesAllFeatures() {
        ArrayList<Plane> planes = new ArrayList<>();
        for (Plane.Names planeName : Plane.Names.values()) {
            ArrayList<Feature> features = new ArrayList<>();
            for (Feature.Names featureName : Feature.Names.values()) {
                features.add(new Feature(featureName));
            }
            planes.add(new Plane(planeName, features));
        }
        return planes;
    }

    private ArrayList<Plane> getExamplePlanesExampleFeatures() {
        ArrayList<Plane> planes = new ArrayList<>();
        ArrayList<Feature> features = new ArrayList<>();
//        features.add(new Feature(Feature.Names.Mean));
        features.add(new Feature(Feature.Names.Gabor_d0r0));
        features.add(new Feature(Feature.Names.Gabor_d0r1));
        features.add(new Feature(Feature.Names.Gabor_d0r2));
        features.add(new Feature(Feature.Names.Gabor_d1r0));
        features.add(new Feature(Feature.Names.Gabor_d1r1));
        features.add(new Feature(Feature.Names.Gabor_d1r2));
        features.add(new Feature(Feature.Names.Gabor_d2r0));
        features.add(new Feature(Feature.Names.Gabor_d2r1));
        features.add(new Feature(Feature.Names.Gabor_d2r2));
        features.add(new Feature(Feature.Names.Gabor_d3r0));
        features.add(new Feature(Feature.Names.Gabor_d3r1));
        features.add(new Feature(Feature.Names.Gabor_d3r2));
        
//        planes.add(new Plane(Plane.Names.Hue, features));
        planes.add(new Plane(Plane.Names.Red, features));
//        planes.add(new Plane(Plane.Names.Value, features));
        return planes;
    }

    private ArrayList<Sample> collectSamples(BufferedImage img, int tileSize) {
        ArrayList<Sample> samples = new ArrayList<>();
        int idx = 0;
        for (int x = 0; x < img.getWidth() - tileSize; x += tileSize) {
            for (int y = 0; y < img.getHeight() - tileSize; y += tileSize) {
                BufferedImage temp = img.getSubimage(x, y, tileSize, tileSize);
                BufferedImage sampleImg = new BufferedImage(tileSize, tileSize, BufferedImage.TYPE_3BYTE_BGR);
                sampleImg.getGraphics().drawImage(temp, 0, 0, null);
                samples.add(new Sample(idx++, x, y, sampleImg));
            }
        }
        return samples;
    }

    private BufferedImage loadImage(File f) {
        try {
            BufferedImage imgIn = ImageIO.read(f);
            return imgIn;
        } catch (IOException ex) {
            System.err.println("Failed to load the input image " + f.getAbsolutePath());
            Logger.getLogger(FeatureGeneratorJOCL_Example.class.getName()).log(Level.SEVERE, null, ex);
            System.exit(-1);
            return null;
        }
    }

    class Stat{
        double min = Double.POSITIVE_INFINITY, 
               max = Double.NEGATIVE_INFINITY, 
               std = 0, 
               mean = 0,
               sum = 0;
    }
    
    private Stat[] getStats(double results[][]){
        int numelFeatures = results[0].length;
        Stat stats[] = new Stat[numelFeatures];
        for (int i = 0; i < stats.length; i++){
            stats[i] = new Stat();
        }

        for (double[] result : results) {
            for (int f = 0; f < result.length; f++) {
                double r = result[f];
                Stat stat = stats[f];
                if (r == r) {
                    stat.sum += r;
                    stat.min = Math.min(stat.min, r);
                    stat.max = Math.max(stat.max, r);
                }
            }
        }

        for (int f = 0; f < results[0].length; f++) {
            Stat stat = stats[f];
            stat.mean = stat.sum / results.length;
        }

        for (double[] result : results) {
            for (int f = 0; f < result.length; f++) {
                double r = result[f];
                Stat stat = stats[f];
                if (r == r) {
                    stat.std += ((r - stat.mean) * (r - stat.mean));
                }
            }
        }

        for (int f = 0; f < results[0].length; f++) {
            Stat stat = stats[f];
            stat.std = Math.sqrt(stat.std / results.length);
        }
        return stats;
    }
    
    public FeatureGeneratorJOCL_Example() {
        initComponents();

//        kernelPlotter();
//        if (true){
//            return;
//        }
        
        // Load the target image
        final BufferedImage imgTarget = loadImage(new File("C:\\Users\\benbryan\\Desktop\\wolf-furry_00339248.jpg"));
        final int tileSize = 20;
        
        ArrayList<Sample> samples = collectSamples(imgTarget, tileSize);
        
        // Create a new feature generator using on a specific platform and device idx
        FeatureGeneratorJOCL generator = new FeatureGeneratorJOCL(0, 0);

        //Example feature set options
        if (true){
            // Generate only a few features
            generator.setPlanes(getExamplePlanesExampleFeatures());   
        } else {
            // Generate all possible features
            generator.setPlanes(getAllPlanesAllFeatures());
        }
        
        BufferedImage sampleImagesAll[] = new BufferedImage[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            sampleImagesAll[i] = samples.get(i).img;
        }
        double results[][];
        try {
            results = generator.getFeaturesForImages(sampleImagesAll);
        } catch (Throwable ex) {
            Logger.getLogger(FeatureGeneratorJOCL_Example.class.getName()).log(Level.SEVERE, null, ex);
            return;
        } 

        Stat[] stats = getStats(results);

        String[] featureNames = generator.getFeatureNames();
        int numelFeatures = featureNames.length;        
        for (int f = 0; f < results[0].length; f++) {
            System.out.println(featureNames[f] + " Mean=" + String.valueOf(stats[f].mean) + " Std=" + String.valueOf(stats[f].std));
        }
        
        double ll[] = new double[numelFeatures];
        double ul[] = new double[numelFeatures];
        
        // Scaling options
        if (true){
            // Statistical
            for (int f = 0; f < results[0].length; f++) {
                Stat stat = stats[f];
                ll[f] = stat.mean - 3 * stat.std;
                ul[f] = stat.mean + 3 * stat.std;
            }
        } else {
            // Hard min max
            for (int f = 0; f < results[0].length; f++) {
                Stat stat = stats[f];
                ll[f] = stat.min;
                ul[f] = stat.max;
            }
        }
        
        final BufferedImage resultImgs[] = new BufferedImage[numelFeatures];
        Graphics graphics[] = new Graphics[numelFeatures];
        for (int f = 0; f < resultImgs.length; f++) {
            resultImgs[f] = new BufferedImage(imgTarget.getWidth(), imgTarget.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
            graphics[f] = resultImgs[f].getGraphics();
        }

        addTab(imgTarget, "Original");
        for (int i = 0; i < resultImgs.length; i++) {
            addTab(resultImgs[i], featureNames[i]);
        }
        
        for (int s = 0; s < results.length; s++) {
            for (int f = 0; f < results[s].length; f++) {
                double fVal = results[s][f];
                Sample sample = samples.get(s);
                fVal = (fVal - ll[f]) / (ul[f] - ll[f]);
                fVal = Math.min(1, fVal);
                fVal = Math.max(0, fVal);
                float fValFloat = (float) fVal;
                graphics[f].setColor(new Color(fValFloat, fValFloat, fValFloat));
                graphics[f].fillRect(sample.x, sample.y, tileSize, tileSize);
            }
        }        
    }

    private void addTab(final BufferedImage img, String title) {
        JPanel panel = new JPanel() {
            @Override
            public void paint(Graphics g) {
                super.paint(g);
                g.drawImage(img, 0, 0, img.getWidth(), img.getHeight(), null);
            }
        };
        panel.setSize(img.getWidth(), img.getHeight());

        JScrollPane jScrollPane = new javax.swing.JScrollPane();
        GroupLayout jPanel1Layout = new GroupLayout(panel);
        panel.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
                jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addGap(0, panel.getWidth(), Short.MAX_VALUE)
        );
        jPanel1Layout.setVerticalGroup(
                jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addGap(0, panel.getHeight(), Short.MAX_VALUE)
        );
        jScrollPane.setViewportView(panel);
        jTabbedPane.addTab(title, jScrollPane);
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jTabbedPane = new javax.swing.JTabbedPane();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jTabbedPane, javax.swing.GroupLayout.DEFAULT_SIZE, 629, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jTabbedPane, javax.swing.GroupLayout.DEFAULT_SIZE, 492, Short.MAX_VALUE)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(FeatureGeneratorJOCL_Example.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(FeatureGeneratorJOCL_Example.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(FeatureGeneratorJOCL_Example.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(FeatureGeneratorJOCL_Example.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new FeatureGeneratorJOCL_Example().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTabbedPane jTabbedPane;
    // End of variables declaration//GEN-END:variables
}
