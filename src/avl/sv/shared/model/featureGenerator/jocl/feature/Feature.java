package avl.sv.shared.model.featureGenerator.jocl.feature;


public class Feature {
    public float[] data;
    public final Names featureName;
    
    public enum Names {
        Mean, 
        Moment2, 
        Moment3, 
        ContrastR, 
        ContrastD, 
        ContrastMagnitude, 
        ContrastSum, 
        HomogeneityR, 
        HomogeneityD, 
        HomogeneityMagnitude, 
        HomogeneitySum, 
        CorrelationR, 
        CorrelationD, 
        CorrelationMagnitude, 
        CorrelationSum,
        Skewness, 
        Kurtosis,
        Gabor_d0r0,
        Gabor_d0r1,
        Gabor_d0r2,
        Gabor_d1r0,
        Gabor_d1r1,
        Gabor_d1r2,
        Gabor_d2r0,
        Gabor_d2r1,
        Gabor_d2r2,
        Gabor_d3r0,
        Gabor_d3r1,
        Gabor_d3r2;
    }
    
    public Feature(Names featureName) {
        this.featureName = featureName;
    }
    
    @Override
    public String toString() {
        return featureName.name();
    }    
}
