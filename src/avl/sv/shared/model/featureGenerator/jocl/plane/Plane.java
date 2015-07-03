package avl.sv.shared.model.featureGenerator.jocl.plane;

import avl.sv.shared.model.featureGenerator.jocl.feature.Feature;
import java.util.ArrayList;

public class Plane {
    public enum Names {
        Red, Green, Blue, Gray, Hue, Saturation, Value
    }
    
    public final Names planeName;
    public final ArrayList<Feature> features;
    
    public Plane(Names planeName, ArrayList<Feature> features) {
        this.planeName = planeName;
        this.features = features;
    }

    @Override
    public String toString() {
        return planeName.name();
    }
    
    
}
