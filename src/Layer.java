public class Layer implements NetworkConstants {

    private Perceptron[] neurons;

    public Layer(int numberOfPerceptrons, int dataLength) {
        neurons = new Perceptron[numberOfPerceptrons];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Perceptron(dataLength, TARGET, LEARNING_RATE, EPOCHS, POWER, true); //data length for networks is just length of activation vector
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
}
