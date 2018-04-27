const tf = require("@tensorflow/tfjs");

const x = tf.variable(tf.scalar(1));
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

// y = a * x ^ 3 + b * x ^ 2 + c * x + d

function predict(x) {
  return tf.tidy(() => {
    return a
      .mul(x.pow(tf.scalar(3))) // a * x ^ 3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x))
      .add(d);
  });
}

const y = predict(x);
a.print();
b.print();
c.print();
d.print();
x.print();
y.print();
