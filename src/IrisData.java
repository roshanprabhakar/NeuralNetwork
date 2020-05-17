public class IrisData {

    private Vector input;
    private Vector output;

    public IrisData(Vector input, Vector output) {
        this.input = input;
        this.output = output;
    }

    public Vector getInput() {
        return input;
    }

    public Vector getOutput() {
        return output;
    }

    public String toString() {
        return "{" + input + "}[" + output + "]";
    }
}
