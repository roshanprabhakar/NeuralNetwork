public class NetworkData {

    private Vector input;
    private Vector correct;

    public NetworkData(Vector input, Vector correct) {
        this.input = input;
        this.correct = correct;
    }

    public Vector getInput() {
        return input;
    }

    public void setInput(Vector input) {
        this.input = input;
    }

    public Vector getCorrect() {
        return correct;
    }

    public void setCorrect(Vector correct) {
        this.correct = correct;
    }
}
