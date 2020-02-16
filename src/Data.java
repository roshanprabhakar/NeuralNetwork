public class Data {

    private Vector values;
    private String id;

    public Data(String id, Vector values) {
        this.values = values;
        this.id = id;
    }

    public Vector getVector() {
        return values;
    }

    public String getId() {
        return id;
    }

    public void setVector(Vector v) {
        this.values = v;
    }

    public void setId(String id) {
        this.id = id;
    }

//    public String toString() {
//        return "id: " + id + ", " + "vector: " + values;
//    }

    public int size() {
        return values.length();
    }

    public String toString() {
        return values.toString();
    }
}
