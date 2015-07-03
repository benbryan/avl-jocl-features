package avl.sv.shared.model.featureGenerator.jocl.feature;

public class GaborKernal {
    private final float real[], imag[];
    private final int kernalSize;

    public float[] getReal() {
        return real;
    }

    public float[] getImag() {
        return imag;
    }

    public int getKernalSize() {
        return kernalSize;
    }
    
    public GaborKernal( int kernalSize, 
                        double waveLength, 
                        double bandwidth, 
                        double aspectRatio, 
                        double phaseOffset, 
                        double orientation) {
        double sigma = calculateSigma(waveLength, bandwidth);
        real = new float[kernalSize * kernalSize];
        imag = new float[kernalSize * kernalSize];
        this.kernalSize = kernalSize;
        for (int k = 0, x = -kernalSize / 2; x <= kernalSize / 2; x++) {
            for (int y = -kernalSize / 2; y <= kernalSize / 2; y++) {
                double x1 = x * Math.cos(orientation) + y * Math.sin(orientation);
                double y1 = -x * Math.sin(orientation) + y * Math.cos(orientation);
                real[k] += (float) (gaborReal(x1, y1, sigma, aspectRatio, waveLength, phaseOffset));
                imag[k] += (float) (gaborImag(x1, y1, sigma, aspectRatio, waveLength, phaseOffset));
                k++;
            }
        }
        double sumReal = 0, sumImag = 0;
        for (int i = 0; i < kernalSize; i++) {
            for (int j = 0; j < kernalSize; j++) {
                sumReal += real[i * kernalSize + j];
                sumImag += imag[i * kernalSize + j];
            }
        }
        sumImag /= (kernalSize*kernalSize);
        sumReal /= (kernalSize*kernalSize);
        for (int i = 0; i < kernalSize; i++) {
            for (int j = 0; j < kernalSize; j++) {
                real[i * kernalSize + j] -= sumReal;
                imag[i * kernalSize + j] -= sumImag;
            }
        }
    }

    private static double calculateSigma(double waveLength, double bandwidth) {
        return waveLength * Math.sqrt(Math.log(2) / 2) * (Math.pow(2, bandwidth) + 1) / ((Math.pow(2, bandwidth) - 1) * Math.PI);
    }

    private static double gaborReal(double x, double y, double sigma, double aspectRatio, double waveLength, double phaseOffset) {
        return Math.exp(-(Math.pow(x / sigma, 2) + Math.pow(y * aspectRatio / sigma, 2)) / 2) * Math.cos(2 * Math.PI * x / waveLength + phaseOffset);
    }
    
    private static double gaborImag(double x, double y, double sigma, double aspectRatio, double waveLength, double phaseOffset) {
        return Math.exp(-(Math.pow(x / sigma, 2) + Math.pow(y * aspectRatio / sigma, 2)) / 2) * Math.sin(2 * Math.PI * x / waveLength + phaseOffset);
    }
    
}
