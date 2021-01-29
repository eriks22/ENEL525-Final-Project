**ENEL 525 F2020 -- Final Project**

**Erik Skoronski**

**December 16th, 2020**

Introduction
============

The COVID-19 pandemic has defined an unprecedented set of problems for
researchers and scientists to solve. In this array of problems, the task
of testing and correctly identifying positive and negative cases of
COVID-19 has been of utmost importance.

This project aims to use the available technologies and techniques of
machine learning to develop a network that can identify positive
COVID-19 cases. This is done using a publicly-available dataset
containing various measures of blood markers, ages, and SARS-CoV-2 test
results of different patients. A summary of the information provided and
the data given is has been presented in follows in Table 1.

**Table 1: Blood Count Information Input Data**

!(media/image1.emf){width="6.5in" height="4.548611111111111in"}

The task at hand is to design, configure, train, and test a machine
learning architecture that can accurately and reliably predict positive
COVID-19 cases using the blood count information as input data.

Methodology
===========

The inputs of the problem include all columns in the given dataset. The
inputs for the network are the same as above but will exclude column 0
(the row index of the original dataset) as this data is independent of
the blood data or test results.

All the blood count data have been standardized to have a mean of 0 and
a standard deviation of 1. Since this data is already appropriate to use
to train a neural network, no cleaning of this data will be done. Since
the ages are delivered as percentiles (ranging from 2-28), this input
would proportionally have a much larger effect on the network and
potentially skew results. Therefore, I chose to scale the age data
between values of 0 and 1 to restrict the values into a range that would
affect the weights less.

**Input data and corresponding value ranges:**

![](media/image2.emf){width="6.5in" height="3.0166666666666666in"}

The target of the project is to correctly identify positive and negative
cases, so the output of the network relies on a binary decision. As
mentioned in chapter 22 of the course textbook^\[1\]^, it can be more
accurate to create two dedicated output neurons for each case (positive
or negative) so two neurons in the output layer were used.

The output layer could then be interpreted as such:

![](media/image3.emf){width="2.7222222222222223in"
height="0.9027777777777778in"}

Network Design
--------------

To design the network I first had to decide on a learning algorithm. As
the data provided was continuous, and therefore would cause varying
levels of precision and error in the results, I decided that
backpropagation would be the best algorithm to use.

Following was the network architecture, namely designing the hidden
layers and the transfer functions. The network design was completed by
using the 15 inputs on the input layer and 2 outputs on the output
layer. I determined that the use of two hidden layers would be simple to
implement, allow me to dynamically change the number of neurons in each
layer, and have a level of complexity in the network that was fitting
for the problem.

In the various configurations I used for the network, it appeared as
though having 6 neurons provided a reasonable amount of complexity for
the problem. It also helped avoid too many local minima in finding the
minimum mean-squared-error when repeatedly iterating through the
training data.

Finally, the transfer functions needed to be chosen. In backpropagation,
the hidden layers function best if a non-linear function is used. The
two options suggested in chapter 22 of the textbook \[1\] were the
*logsig* and *tansig* functions. I decided to use tan-sigmoid function
for the hidden and output layers as it allows a

**Formula 1: Hyperbolic tangent sigmoid transfer function:**

$$\text{tansig}\left( n \right) = \ \frac{2}{(1 + e^{- 2n})} - 1$$

$$\frac{d}{\text{dx}}\text{tansig}\left( n \right) = \ f^{'} = (1 - n^{2})$$

Network diagrams:
-----------------

**Figure 1: Neural Network Diagram**

![Diagram Description automatically
generated](media/image4.tiff){width="5.444444444444445in"
height="3.388888888888889in"}

**Figure 2: Hidden Layer Neuron**

![Diagram Description automatically
generated](media/image5.tiff){width="1.7916666666666667in"
height="1.0833333333333333in"}

**Figure 3: Output Layer Neuron**

![Diagram Description automatically
generated](media/image6.tiff){width="1.6666666666666667in"
height="1.0416666666666667in"}

The training was done by using the training data to update the weights
and biases.

Derivation of Backpropagation Learning Rule
-------------------------------------------

Backpropagation derivation (inspired by Chapter 11 of the textbook
\[1\]:

The backpropagation algorithm includes the three following steps:

Where $\overset{⃑}{a} = output$, $\overset{⃑}{e} = error$,
$\overset{⃑}{b} = bias$, and $\alpha = learning\ rate$

Step 1: Propagate inputs through the network

$${\overset{⃑}{a}}^{m + 1} = tansig\left( {\overset{⃑}{W}}^{m + 1}p + {\overset{⃑}{b}}^{m + 1} \right)\text{\ \ \ \ }\mathrm{\text{for\ }}m = 0,\ 1,\ 2,\ \ldots,\ L - 1\ $$

$$\overset{⃑}{a} = {\overset{⃑}{a}}^{L}$$

$$\overset{⃑}{e} = \ t - a$$

Step 2: Back propagate the sensitivities

$$\frac{\partial\widehat{F}}{{\partial\overset{⃑}{n}}^{L}} = - 2F^{'L}({\overset{⃑}{n}}^{L})\overset{⃑}{e}$$

$$\frac{\partial\widehat{F}}{{\partial\overset{⃑}{n}}^{m}} = F^{'m}\left( {\overset{⃑}{n}}^{m} \right)\left( {\overset{⃑}{W}}^{m + 1} \right)\frac{\partial\widehat{F}}{{\partial\overset{⃑}{n}}^{m + 1}}\text{\ \ }\mathrm{\text{for\ }}m = L - 1,\ L - 2,\ \ldots,\ 2,\ 1\ $$

Step 3: Update the weights and biases

$${\overset{⃑}{W}}^{m}\left( k + 1 \right) = {\overset{⃑}{W}}^{m}\left( k \right) - \alpha\frac{\partial\widehat{F}}{{\partial\overset{⃑}{n}}^{m}}({\overset{⃑}{a}}^{m - 1})$$

$${\overset{⃑}{b}}^{m}\left( k + 1 \right) = {\overset{⃑}{b}}^{m}\left( k \right) - \alpha\frac{\partial\widehat{F}}{{\partial\overset{⃑}{n}}^{m}}$$

Using these equations we can implement the back-propagation algorithm
used for this project.

Training and Testing Scheme
---------------------------

90% of the data was used for training and the remainder 10% of data was
used for testing. To ensure good input data, I randomly distributed the
data until the first 538 entries had roughly the same proportion of
positive tests as the last 60 test points. In the dataset, 13.55% of the
cases tested positive (81 positive / 598 total cases).

To minimize the mean squared error (MSE) of the neural network, I set
the learning rate to 0.01 and the error threshold to 0.02.

Results and Discussion
======================

The network required a considerable amount of altering to get the right
balance of complexity and simplicity.

Final results:

**Figure 4: Mean Squared Error at each Iteration**

![Chart Description automatically
generated](media/image7.png){width="5.384933289588801in"
height="2.3663363954505687in"}

**Figure 5: All Output vs Testing and Training Data**

![A picture containing diagram Description automatically
generated](media/image8.png){width="4.3480621172353455in"
height="1.5748031496062993in"}

![A picture containing diagram Description automatically
generated](media/image8.png){width="0.2475929571303587in"
height="1.5748031496062993in"}![A picture containing diagram Description
automatically generated](media/image8.png){width="4.1943908573928255in"
height="1.5748031496062993in"}

![A picture containing diagram Description automatically
generated](media/image8.png){width="0.2475929571303587in"
height="1.5748031496062993in"}![A picture containing diagram Description
automatically generated](media/image8.png){width="4.282087707786527in"
height="1.5748031496062993in"}

**Figure 6: Output vs. Test Data**

![Chart, histogram Description automatically
generated](media/image9.png){width="3.3472222222222223in"
height="2.6805555555555554in"}

*Note: Chart above only shows 2^nd^ neuron in the output layer as output
neurons are highly inversely-correlated *

Conclusion
==========

The mean-squared-error of the neural network with the training data was
able to be reduced down to 0.02, however, the mean-squared-error of the
test data was 0.0343. Based on the output data (figure 6), the network
cannot correctly distinguish between positive and negative results.

This may be mainly due to the training data. As there is a different
proportion of positive to negative cases in the network, the network is
mainly trained to identify negative cases. If more data were available,
or if potentially alternative training methods were explored, then
potentially the error could be reduced.

As well, the use of backpropagation may have been inappropriate for this
task. Other methods, such as Decision Tree or Random Forest modelling
^\[2\]^ may be more suitable for the problem. These methods, however,
were beyond the scope of this course.

Regardless of the results of the network, valuable skills were learned
about the real-world application of machine learning techniques. As
well, I gained valuable knowledge about the importance of the design of
neural network architecture for machine learning.

References
==========

\[1\] Hagan, M. T., Demuth, H. B., Beale, M. H., & Jesús, O. (2014).
Neural network design (2nd ed.). Stillwater, Oklahoma: Martin Hagan.

\[2\] Brinati, D., Campagner, A., Ferrari, D., Locatelli, M., Banfi, G.,
& Cabitza, F. (2020). Detection of COVID-19 Infection from Routine Blood
Exams with Machine Learning: A Feasibility Study. Journal of medical
systems, 44(8), 135. https://doi.org/10.1007/s10916-020-01597-4

Appendix
========

MATLAB code: (attached as .m file)
