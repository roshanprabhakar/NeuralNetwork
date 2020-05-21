public class Vector {

    private double[] vector;

    public Vector(double[] vector) {
        this.vector = vector;
    }

    public Vector(int i) {
        this.vector = new double[i];
    }

    public Vector(int i, boolean random) {
        this.vector = new double[i];
        for (int j = 0; j < vector.length; j++) {
            vector[j] = Math.random() * 1;
        }
    }


    public int length() {
        return vector.length;
    }

    public double get(int i) {
        return vector[i];
    }

    public void set(int index, double value) {
        vector[index] = value;
    }


    public Vector multiplyScalar(double scalar) {
        Vector copy = this.copy();
        return multiplyScalar(scalar, copy);
    }

    public static Vector multiplyScalar(double scalar, Vector v) {
        Vector out = v.copy();
        for (int i = 0; i < v.length(); i++) {
            out.set(i, out.get(i) * scalar);
        }
        return out;
    }

    public Vector add(Vector other) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] += other.get(i);
        }
        return this;
    }

    public void concat(Vector other) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] += other.get(i);
        }
    }

    public static Vector add(Vector v, Vector p) {
        Vector out = new Vector(v.length());
        for (int i = 0; i < v.length(); i++) {
            out.set(i, v.get(i) + p.get(i));
        }
        return out;
    }

    public Vector subtract(Vector other) {
        this.add(other.multiplyScalar(-1));
        return this;
    }

    public double dotProduct(Vector other) {
        return dotProduct(this, other);
    }

    public static double dotProduct(Vector v, Vector p) {
        double sum = 0;
        String out = "";
        for (int i = 0; i < v.length(); i++) {
            out += v.get(i) + " * " + p.get(i);
            if (i != v.length() - 1) out += " + ";
            sum += v.get(i) * p.get(i);
        }
//        System.out.println(out + " = " + sum);
        return sum;
    }

    public static double multiply(Vector v, Vector p) {
        return dotProduct(v, p);
    }

    public Vector expand(int power) {
        Vector out = new Vector(this.length() * power);
        for (int i = 0; i < length(); i++) {
            for (int pow = 1; pow <= power; pow++) {
                out.set(i * power + pow - 1, Math.pow(this.get(i), pow));
            }
        }

        return out;
    }

    public double loss(Vector correct) {
        if (correct.length() != this.length()) return Double.MAX_VALUE;
        double loss = 0;
        for (int i = 0; i < this.length(); i++) {
            loss += (correct.get(i) - this.get(i)) * (correct.get(i) - this.get(i));
        }
        return loss * (0.5);
    }

    public Vector copy() {
        return new Vector(vector.clone());
    }

    public double[] getVector() {
        return vector;
    }

//    public String toString() {
//        return Arrays.toString(vector);
//    }

    public String toString() {
        StringBuilder out = new StringBuilder("(");
        for (int i = 0; i < vector.length; i++) {
            out.append(Data.round(vector[i], 1000));
//            out.append(vector[i]);
            if (i != vector.length - 1) out.append(", ");
        }
        out.append(")");
        return out.toString();
    }

    public Vector getNetworkOutputVector() {
        double max = Integer.MIN_VALUE;
        for (double value : this.getVector()) {
            if (value > max) max = value;
        }
        Vector out = new Vector(this.length());
        for (int i = 0; i < this.length(); i++) {
            if (this.get(i) == max) {
                out.set(i, 1);
            }
        }
        return out;
    }

    public boolean equals(Vector other) {
        assert length() == other.length();
        for (int i = 0; i < other.length(); i++) {
            if (get(i) != other.get(i)) return false;
        }
        return true;
    }

}
