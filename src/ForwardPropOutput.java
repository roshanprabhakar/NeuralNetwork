public class ForwardPropOutput {

    private Vector resultant;
    private Vector[] intermediaryMatrix;

    public ForwardPropOutput(Vector resultant, Vector[] intermediaryMatrix) {
        this.resultant = resultant;
        this.intermediaryMatrix = intermediaryMatrix;
    }

    public Vector getResultant() {
        return resultant;
    }

    public Vector[] getIntermediaryMatrix() {
        return intermediaryMatrix;
    }
}
