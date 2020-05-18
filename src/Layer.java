public class Layer implements NetworkConstants {

    private Perceptron[] neurons;

    public Layer(int numberOfPerceptrons, int dataLength) {
        neurons = new Perceptron[numberOfPerceptrons];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Perceptron(dataLength, TARGET, LEARNING_RATE, EPOCHS, POWER, false); //data length for networks is just length of activation vector
        }
    }

    public Perceptron get(int i) {
        return neurons[i];
    }

    public Vector activations(Vector previous) { //activations from previous layer
        Vector activations = new Vector(neurons.length);
        for (int i = 0; i < neurons.length; i++) {
            activations.set(i, neurons[i].guess(previous));
        }
        return activations;
    }

    public int length() {
        return neurons.length;
    }

    public String toString() {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < neurons.length; i++) {
            out.append("[").append(neurons[i].getWeights()).append("]{").append(Data.round(neurons[i].getBias(), 100)).append("}").append("   ");
        }
        return out.toString();
    }
}
