package avl.sv.shared.model.featureGenerator.jocl.feature;

import java.util.ArrayList;
import org.jocl.cl_event;
import org.jocl.cl_mem;

public class CL_Feature {
    public Feature.Names name;
    public cl_mem cl_data;
    public cl_event cl_event = new cl_event();
    public final ArrayList<cl_event> waitForEvents = new ArrayList<>();
}
