---
layout: distill
title: Pushing Meta-Continual Learning Algorithms to the Limit
description: Meta-continual learning algorithms should be able to handle tasks with extended data streams compared to the traditional deep learning setting. These algorithms have not been applied to settings with extreme data streams, such as classification tasks with 1,000 classes, nor have they been compared to traditional continual learning algorithms. We compare meta-continual learning to continual learning and we find that meta-continual learning scales better than continual learning.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous



# must be the exact same name as your blogpost
bibliography: 2026-04-27-pushing-meta-cl-methods.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Background
    subsections:
      - name: Continual Learning
      - name: REMIND Algorithm
      - name: Meta-Learning
      - name: OML Algorithm
      - name: GeMCL Algorithm
  - name: Experiments
    subsections:
      - name: Datasets
      - name: Method
      - name: Implementation Details
      - name: CASIA Experiment
      - name: Omniglot Experiment
  - name: Results and Analysis
    subsections:
    - name: CASIA Results and Analysis
    - name: Omniglot Results and Analysis
  - name: Discussion
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction
In traditional deep learning, models are often trained on the assumption that the data used to train and test the model is independently and identically distributed (IID) and the data distribution is stationary.
In real-world settings, the assumption of stationarity is frequently violated, particularly in domains like healthcare, where the continuous emergence of novel diseases and treatment protocols leads to inherently non-stationary data distributions <d-cite key="singh_class-incremental_2023"></d-cite>.
For instance, imagine a robot deployed to vacuum clean a house; it would be challenging to pre-program the robot to handle all possible scenarios in all households.
Ideally, deployed models should be capable of adapting to evolving data distributions and support continual learning to acquire new skills post-deployment.
Naive fine-tuning can lead to updates where the model's parameters overwrite previously acquired knowledge, thereby impairing its ability to perform earlier-learned tasks.
This phenomenon is known as catastrophic forgetting (CF) <d-cite key="mccloskey_catastrophic_1989"></d-cite>.
expensiveTraining the model from scratch (known as the Offline approach) every time new data arrives is computationally expensive and impractical <d-cite key="verwimp_continual_2023"></d-cite>, especially if the new data is much smaller than the original data used to train the model.

Continual Learning (CL) is one approach to tackling CF.
CL is about learning new tasks sequentially without forgetting how to perform earlier tasks, ideally without requiring access to data from earlier tasks.
The goal of CL is to ensure that the update of the model is done in a computationally efficient way and that the continual learner performs as if it were trained with all the available data.
However, Harun et al. <d-cite key="harun_how_2023"></d-cite> showed that there are CL methods that are more computationally expensive than the Offline approach. They also mention that CL algorithms tend to focus on mitigating CF at the cost of computational efficiency.

Meta-learning is a field focused on training models to learn how to learn. Meta-learning algorithms should ideally allow models to generalise to out-of-distribution tasks and efficiently adapt to new tasks. The intersection between CL and meta-learning, known as meta-continual learning (meta-CL), is about learning to continually learn. In theory, meta-CL algorithms should be more sample efficient than CL methods and perform better on tasks with extended data streams, such as classification tasks with 1,000 classes.

However, existing meta-CL methods such as Online-aware Meta-learning (OML) <d-cite key="khurram_javed_meta-learning_2019"></d-cite> have only been shown to continually learn classification tasks with a few hundred classes.
Generative meta-continual learning (GeMCL) <d-cite key="mohammadamin_banayeeanzade_generative_2021"></d-cite> is another meta-CL algorithm that has been shown to do this and is theoretically immune to CF.

Neither GeMCL nor OML has yet been tested on tasks with 1,000 classes, nor have they been compared to actual CL algorithms at scale.
These algorithms have also not been tested at scale on other datasets.
Meta-CL and CL algorithms should be able to handle extended data streams that are significantly longer than those found in traditional deep-learning scenarios.

The purpose of this blog is to compare the meta-CL algorithms GeMCL and OML to the CL algorithm REMIND <d-cite key="vedaldi_remind_2020"></d-cite> and the Offline approach on the CASIA Chinese Handwriting Database (CASIA) <d-cite key="liu_casia_2011"></d-cite> on 1,000-way-10-shot classification tasks.
Furthermore, since OML and GeMCL are meta-learning algorithms, they should be able to generalise to other datasets. We also aim to test this by training OML and GeMCL on the CASIA dataset and see if they generalise to the Omniglot dataset <d-cite key="lake_human-level_2015"></d-cite> on 1,000-way-10-shot classification tasks.

We aim to see whether there are any benefits to using meta-learning approaches or whether one should simply stick to standard CL methods at scale. The roadmap to our destination is as follows: we first begin by explaining what CL is and introduce the REMIND algorithm, which has been shown to be computationally efficient compared to other CL algorithms <d-cite key="harun_how_2023"></d-cite>. Our next stop covers meta-learning and the meta-CL algorithms: OML and GeMCL. Following that, we conduct experiments on the 1,000-way-10-shot tasks and present our findings and thoughts.

## Background
### Continual Learning
<!-- _Need visualisations of the distribution changes when we add more samples. Need to define CF as well and state why it happens. Mention the Desiderata of CL._ -->

We present CL in the context of supervised classification, though it extends to other domains.
Assume we have the following training set: $$\mathcal{D}_{t}=\{\mathcal{X}_{t}, \mathcal{Y}_{t}\}$$. Here $$t$$ denotes the task, $$\mathcal{X}_{t}$$ is the input data and $$\mathcal{Y}_{t}$$ are the labels.

Before we look at the three main scenarios in continual learning, it is important that we make an assumption. Going forward, we assume all task boundaries are clearly defined. What we mean by this is that there is no overlap in the labels between different tasks. More formally, we assume

$$
\mathcal{Y}_{i} \cap \mathcal{Y}_{j} = \emptyset \quad \text{for any task } i \text{ and }j.
$$

Now we can move on to the three main scenarios in CL <d-cite key="gido_m_van_de_ven_three_2019"></d-cite>: task-incremental learning (task-IL), domain-incremental learning (domain-IL) and class-incremental learning (class-IL) <d-footnote>There are more scenarios in CL. For more information on the different types of scenarios, we recommend the survey by Wang et al. <d-cite key="wang_comprehensive_2024"></d-cite>.</d-footnote>.
In task-IL, the task ID is provided during both training and testing. This means one can train their model with task-specific components. As van de Ven et al. <d-cite key="gido_m_van_de_ven_three_2019"></d-cite> point out, task-IL is the easiest scenario in CL. Domain-IL differs in that the task ID is not provided during testing; however, the model does not have to infer what task it is solving. Class-IL also does not have the task ID at training or test time; however, it must infer what task it is solving. To better explain these scenarios, we will take a look at the following example based on the example provided by van de Ven et al. <d-cite key="gido_m_van_de_ven_three_2019"></d-cite>.

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/task_visualization_separated_seed42.png" class="img-fluid" %}
<div class="caption">
    Figure 1. An example of the characters from the CASIA database broken up into tasks. Example based off <d-cite key="gido_m_van_de_ven_three_2019"></d-cite>.
</div>

Figure 1 shows 3 tasks using 6 characters from the CASIA database. Each task in the figure contains different characters, but within each task they are labelled as Class 1 or Class 2.
In the task-IL scenario, the model is given a task and the task ID and must predict either Class 1 or Class 2.
For domain-IL, the model is given a character but is not told the task ID, yet must still distinguish between Class 1 and Class 2.
The class-IL scenario also withholds the task ID; however, the model must now output the specific character identity (e.g., Character 1 through Character 6), rather than simply outputting Class 1 or Class 2.
In the work, we focus on the class-IL.

CL aims to mitigate CF, which occurs when data is learned sequentially instead of simultaneously. For example, if Tasks 1 through 3 from Figure 1 are learned simultaneously, we expect the model to perform well on all three tasks.

However, if we naively learn the three tasks sequentially, starting with Task 1 and ending with Task 3, CF often occurs because each of the three tasks have different minima.
This occurs because the weights learned for Task 1 are overridden as the model updates its parameters to optimise for Task 2. Similarly the parameters learned for Task 2 are overridden when we learn Task 3. If all tasks are presented simultaneously, there is a tug-of-war on the parameters, whereby the gradient from each task pulls the model toward a minimum that yields reasonable performance across all three tasks <d-cite key="raia_hadsell_embracing_2020"></d-cite>.

Usually in CL, the model cannot access the data from previously seen tasks. One category of CL methods known as replay-based methods aims to tackle this by storing a subset of past examples shown during training. Typically, as the learner learns a task, examples from that task are stored in a replay buffer. The model then replays examples stored while learning new classes.

The replay-based method we will focus on is the REMIND algorithm, as it has been shown to be more computationally efficient than other replay-based methods <d-cite key="harun_how_2023"></d-cite>.

###  REMIND Algorithm
REMIND (**re**play using **m**emory **ind**exing) <d-cite key="vedaldi_remind_2020"></d-cite> is inspired by biological replay.
Instead of storing raw feature representations of the input in its replay buffer, it stores the compressed representation. This allows REMIND to store more 
representations in its replay buffer compared to other replay-based methods <d-cite key="vedaldi_remind_2020"></d-cite>.
<!-- To compress the features REMIND makes use of product quantisation (PQ) <d-cite key="jegou_product_2011"></d-cite>. It is a lossy compression method. -->

#### Product Quantisation

To compress the features REMIND makes use of product quantisation (PQ) <d-cite key="jegou_product_2011"></d-cite>. It is a lossy compression method.
PQ works by taking a vector of dimension $$D$$ and dividing it into $$m$$ sub-vectors. $$D$$ must be divisible by $$m$$, meaning each sub-vector will have a dimension of $$D/m$$.

Each sub-vector belongs to its own independent subspace, and we process these subspaces separately. We make use of the $k$-means algorithm to cluster the sub-vectors within each specific subspace. This means that if we select $$k$$ centroids, each independent subspace will calculate and store its own unique set of $$k$$ centroids. This collection of centroids for a subspace is often referred to as a codebook.

During the encoding process, each sub-vector is assigned the ID of the closest centroid from its respective subspace. Therefore, the original vector of dimension $$D$$ is compressed into a vector of IDs with dimension $$m$$. This process of compressing the vector by assigning centroid IDs is known as encoding.

The codebook is then used for reconstruction. To decode the vector, we look up each centroid ID in the compressed vector and replace it with the corresponding centroid coordinates from that subspace. Concatenating these $$m$$ parts yields a reconstructed vector of dimension $$D$$. For more details on PQ see Efimov <d-cite key="efimov_similarity_2023"></d-cite>.

#### REMIND's Training Procedure

The REMIND algorithm works by freezing certain parts of the neural network during continual learning. We freeze the first few layers of the network, known as the frozen layers (or encoder), and only update the later layers, known as the plastic layers. Figure 2 illustrates the flow of the REMIND algorithm. In the figure, our frozen layers form a CNN encoder, and the plastic layers are the MLP layers of the neural network.

The encoder takes in an image, $$x_i$$, and outputs a feature representation, $$z_i$$. The representation $$z_i$$ is then encoded using PQ. We then sample $$r$$ encoded examples from the replay buffer, including their respective labels.

We then form a batch comprising the current encoded $$z_i$$ (and its respective label) and the $$r$$ sampled examples. We decode this batch to form the decoded batch $$\mathcal{Z}$$ with $$r+1$$ samples. We use this decoded batch $$\mathcal{Z}$$ and the corresponding labels $$Y$$ to update the MLP layers via SGD.

After updating the MLP layers, we store the encoded $$z_i$$ and its label in the replay buffer. When the buffer is full, we randomly remove one sample from the most represented class to make space. This process continues until all classes are learned sequentially.

During testing, we take the test set, $$\tilde{\mathcal{X}}$$, and pass it through the encoder to obtain the representation $$\tilde{\mathcal{Z}}$$. We then encode and decode the representation using PQ before passing it through the MLP to make our predictions.

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/REMIND.png" class="img-fluid" %}
<div class="caption">
    Figure 2. The training procedure of the REMIND algorithm. REMIND takes in an image and passes it through the frozen layers of the encoder (enc) to output feature representations. 
    It is then compressed using PQ and stored in the replay buffer. We then sample $r$ other compressed feature representations from the replay buffer and combine them with the current input to form a batch.
    We take this batch and its corresponding labels to update the plastic layers of the neural network, which are the MLP layers in this case. During testing, the test set is passed through, and we encode and decode the feature representations before passing them through the MLP to make predictions.
</div>

To learn $$J$$ classes, we first take a subset of $$K$$ classes. We use this subset to train the neural network in the standard supervised manner, without using PQ. After pre-training, we freeze the encoder and perform continual learning on the remaining classes. We initialise the PQ model using the pre-training subset and store the encoded samples from these classes in the replay buffer.

The REMIND algorithm also makes use of augmentation during replay to make representations more robust. Random resized crops of the encoded samples are used. Manifold mixup <d-cite key="verma_manifold_2019"></d-cite> is also used to mix features from multiple classes. This works by sampling two batches, $$A$$ and $$B$$, with $$r$$ samples from the replay buffer. We reconstruct (decode) these samples and linearly combine them to form a batch $$C$$ with $$r$$ samples. Finally, we combine $$C$$ with the current input to form the batch for decoding and feed it into the MLP to make predictions.

### Meta-Learning
Modern machine learning often requires large amounts of data and training, limiting how quickly models can adapt to new tasks <d-cite key="huisman_survey_2021, timothy_m_hospedales_meta-learning_2020"></d-cite>.
Meta-learning addresses this challenge by enabling models to learn how to learn. A meta-learner is trained on a distribution of tasks so that it can leverage this experience to quickly adapt to new, unseen tasks. Figure 3 illustrates this process. While training a meta-learner can be computationally expensive, this upfront cost is offset at inference time through rapid adaptation to new tasks.

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/meta-learnin.png" class="img-fluid" %}
<div class="caption">
    Figure 3. An illustration of meta-learning. Adapted from <d-cite key="yu_meta-world_2021"></d-cite>.
</div>

In standard supervised learning, we aim to find parameters $$\mathbf{\theta}$$ such that we minimise the loss of some task $$t$$. The loss of task $$t$$ is given by $$\mathcal{L}_{t}$$.
The training set of task $$t$$ is $$\mathcal{D}_t$$.

In meta-learning, when trying to find the optimal $$\mathbf{\theta}$$, we often provide some knowledge about how to learn or to guide the model in its training phase <d-cite key="timothy_m_hospedales_meta-learning_2020"></d-cite>.
Let us call this model the base-learner, which has parameters $$\theta$$.
Examples of how we provide this knowledge include picking the optimiser, deciding how to initialise the neural network, and choosing hyperparameter values, such as the learning rate.
Collectively, this is known as the meta-knowledge.
The standard supervised problem we aim to solve is:

$$
\theta^* = \underset{\theta}{\arg \min}\,\, \mathcal{L}_{t} \left( \mathcal{D}_t; \theta \right).
$$

However, in meta-learning, we aim to learn parts of the meta-knowledge which we denote as $$\omega$$. Therefore our standard supervised objective is conditioned on $\omega$:

$$
\theta^* = \underset{\theta}{\arg \min}\,\, \mathcal{L}_{t} \left( \mathcal{D}_t; \theta,\omega \right).
$$

To illustrate this, consider a specific application where we aim to learn the learning rate scheduler instead of manually designing it. This meta-knowledge is represented by the parameters of a neural network; in this case, $$\omega$$ is known as the meta-parameters.
This neural network is known as the meta-learner.
In this example, we have a neural network that outputs the learning rate at every step depending on how well the base-learner is doing (i.e., it has memory of the loss from previous updates to determine the appropriate learning rate for the current step).
Ideally, the meta-learner should be able to generalise to different tasks and output the learning rate in such a way that the base-learner solves task $$t$$ more quickly.

Since we generally wish for the learned meta-knowledge to be applied to different tasks, we use a distribution of tasks to train the meta-learner.
This means that there are two stages of learning in meta-learning: the inner-loop and the outer-loop.
The outer-loop is concerned with a distribution of tasks $$\mathcal{P}(T)$$. We then sample a task $$t$$ from $$\mathcal{P}(T)$$ for each outer-loop step.
In the inner-loop, we are concerned with finding $$\theta^*$$ for a specific task; i.e., the inner-loop is the base-learner learning task $$t$$.
Therefore, we aim to find:

$$
\omega^* = \underset{\omega}{\arg \min}\,\, \underbrace{\underset{t \sim \mathcal{P}(T)}{\mathbb{E}}}_{\rm outer-loop} \,\,\underbrace{\mathcal{L}_{\mathcal{D}_t} \left( \mathcal{D}_t; \theta, \omega \right)}_{\rm inner-loop}.
$$

The phase of learning $$\omega^*$$ is known as meta-training, whereas evaluating the learned $$\omega^*$$ is known as meta-testing.

#### $$N$$-way-$$K$$-shot Learning
One common setup for meta-learning in supervised classification is $$N$$-way-$$K$$-shot learning <d-cite key="huisman_survey_2021"><\d-cite>. In $$N$$-way-$$K$$-shot learning, we have $$N$$ classes each with $$K$$ training examples.
Therefore, the training dataset for an $$N$$-way-$$K$$-shot learning episode should have $$N\cdot K$$ total examples. The meta-training stage and meta-testing stage have disjoint labels; i.e., the classes used during meta-training are not the same as the classes used in meta-testing.

The training examples form the support set, while examples used for evaluation form the query set
Thus, the base-learner uses the support set to learn a task, then is evaluated on the query set. The loss on the query set is then used to update $$\omega$$.

The goal of using various tasks or $$N$$-way-$$K$$-shot episodes is to acquire a $$\omega^*$$ that can a) help the base-learning learn a specific skill quickly and b) be used across multiple tasks.

<!-- OML and ANML are regularisation based of meta bench and life learner is a reharsal based method -->

### OML Algorithm
Online aware Meta-Learning (OML) <d-cite key="khurram_javed_meta-learning_2019"></d-cite> is a modification of the model-agnostic meta-learning (MAML) framework <d-cite key="finn_model-agnostic_2017"></d-cite>, adapted for continual learning. MAML is an algorithm that meta-learns a neural network initialisation that allows for the neural network to learn new tasks with just a few gradient updates. However, if you apply MAML's approach directly in the continual learning setting, all parameters are updated, which leads to CF. To address this, OML freezes the earlier layers of the neural network during the inner-loop steps, and only updates the later layers.

Figure 4 illustrates the OML process. The neural network consists of an encoder and MLP layers.
The encoder produces a feature representation of the input and represents our meta-learner; tts parameters $$\omega$$ are therefore the meta-parameters. The MLP parameters $$\theta$$ are updated during inner-loop adaptation and are also called fast weights.

The OML process involves sampling an $$N$$-way-$$K$$-shot episode. In the inner loop, we process each sample from the support set sequentially.
We feed a sample through the encoder, and then the MLP makes a prediction.
We then calculate the loss on that prediction and use SGD to update the fast weights. This continues until we have used each sample in the support set. Once the inner loop is complete, we test our trained MLP using the query set. The loss from that query set is then used to update the meta-parameters (the encoder).

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/oml.png" class="img-fluid" %}
<div class="caption">
    Figure 4. An illustration of the OML process. OML processes each sample from the support set sequentially, using the loss from that sample to update the MLP layers via SGD. After going through the entire support set, we then test the trained MLP on the query set and use the query loss to update the encoder's parameters (the meta-parameters). Adapted from <d-cite key="jaehyeon_son_when_2023"></d-cite>.
</div>

At meta-test time, the encoder is fixed, and we reinitialise the MLP layers as we are learning classes not seen during training.
This reinitialisation happens during meta-training as well: we reinitialise the MLP kayers before starting each new outer-loop episode. This ensures the training setup closely matches meta-testing conditions: where each class encountered is new and the network must learn from scratch using the meta-learned representation.

When performing the meta-update, we back-propagate through the inner loop, requiring second-order derivatives to update the encoder <d-footnote> One can use a first-order approximation instead. </d-footnote>.
Additionally, the inner loop and outer loop typically require different learning rates. For OML, practitioners commonly perform a hyperparameter search for the optimal inner-loop learning rate at meta-test time, as OML is highly sensitive to this value. Ironically, as Irie et al. <d-cite key ="irie_metalearning_2024"></d-cite> observed:

> ... this is one of the
characteristics of hand-crafted learning algorithms that we precisely aim to avoid using learned
learning algorithms.

Furthermore, the OML architecture typically uses a fixed number of output nodes, which implies we assume there will be a maximum number of classes to be continually learned.

### GeMCL Algorithm
Generative Meta-Continual Learning (GeMCL) <d-cite key="mohammadamin_banayeeanzade_generative_2021"></d-cite> uses a generative Bayesian classifier. It follows a similar approach to OML, in that we employ an encoder to meta-learn features; however, GeMCL also models the distribution of each class. We assume each class, $$c$$, is modelled by a Gaussian distribution with mean $$\mu^c$$ and precision $$\lambda^c$$. The class conditional Gaussians form a Gaussian Mixture Model (GMM).

The mean $$\mu^c$$ is also modelled by a Gaussian, for which we assume uninformative priors. The precision is modelled by a Gamma distribution parameterised by $$\alpha$$ (shape) and $$\beta$$ (scale) and serve as the priors for the Gamma distribution and are learnable parameters.

The posterior distributions of the mean $$\mu^c$$  and the precision $$\lambda^c$$ take the form of a Normal-Gamma distribution. This allows us to calculate the posterior parameters using closed-form equations as we learn a class. The predictive distribution is a Student’s t-distribution.

Figure 5 illustrates the GeMCL process. During an $$N$$-way-$$K$$-shot episode, the encoder receives the input and outputs the feature representation of that input. We then use Bayes' Theorem to obtain the posterior parameters of the class-specific distributions using the representation of the input and the prior. The posterior parameters then act as the prior for the next step. This process continues until we have iterated through the entire support set.

Next, we use our model to make predictions on the query set. The encoder and the prior parameters are fixed during this step. We use the predictive distribution to select the class that assigns the highest probability to the observed data point from the query set. This loss is then used to update the encoder, as well as $$\alpha$$ and $$\beta$$<d-footnote> The Gamma distribution priors do not have to be learnable parameters. We can estimate the prior parameters by using the maximum likelihood estimate of the precision of each class's features.</d-footnote>. At the start of a new episode, we initialise the class-specific parameters from the priors.

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/gemcl.png" class="img-fluid" %}
<div class="caption">
    Figure 5. An illustration of the GeMCL process. The encoder encodes the input to output the feature representation of the input. This is then used alongside the prior to update the posterior parameters of the class-specific distribution. After processing the support set, we make use of the predictive distribution to predict the class of the query set. We then use the query set loss to update the encoder. Adapted from <d-cite key="jaehyeon_son_when_2023"></d-cite>.
</div>

In GeMCL, since each class possesses its own set of parameters and is modelled by separate Gaussians, updating the class-specific parameters for class $$A$$ has no effect on the parameters for class $$B$$. The class parameters are isolated. As the encoder is fixed at meta-test time, GeMCL is immune to CF <d-cite key="mohammadamin_banayeeanzade_generative_2021, lee_learning_2024"></d-cite>.
Therefore, the order in which we learn the classes does not matter. We can learn the classes in any sequence and arrive at the same distribution for that class, as the class parameters are isolated. Furthermore, GeMCL makes no assumption regarding the maximum number of classes. To learn a new class, we simply learn the statistics of that class's representations.

## Experiments

Our training process follows the approach of Lee et al. <d-cite key="lee_learning_2024"></d-cite>, which is detailed in the Method section. We adapted their official implementation for our experiments; the code is available at our [anonymised repository](https://anonymous.4open.science/r/SB-MCL-B28E/README.md).

### Datasets
The following datasets are used in these experiments: CASIA Chinese Handwriting Database (CASIA) <d-cite key="liu_casia_2011"></d-cite> and the Omniglot <d-cite key="lake_human-level_2015"></d-cite> dataset. The CASIA dataset consists of 7,356 handwritten characters or classes: 7,185 Chinese characters and 171 symbols (such as punctuation marks and English letters). Since the CASIA dataset is used to meta-train the meta-CL algorithms, we split the classes into meta-training and meta-testing sets. We reserve 5,356 classes for meta-training and 2,000 classes for meta-testing.

The Omniglot dataset consists of 1,623 classes from 50 different alphabets. This dataset is used to evaluate whether the meta-parameters learned on CASIA can generalise to a different dataset. We randomly sampled a fixed set of 1,000 classes to use for these experiments.
<!-- For each class (from both Omniglot and CASIA), we randomly sample `10` shots or examples as the support set for each class.  Should go into training details section.-->
### Method
#### Baselines
We have two non-meta baselines: an Offline learner and REMIND.
The Offline learner represents the upper bound on performance. It does not perform CL; instead, it learns in a standard supervised manner where all data is available from the start. Therefore, the data distribution remains stationary and samples are drawn IID. The Offline learner is trained from scratch for a fixed number of epochs on a 1,000-way-10-shot task, and we record the best performance achieved on the test set. We repeat this for 20 seeds.

REMIND serves as our standard CL baseline. In our experiments, we follow the standard REMIND training protocol. The model is trained on a 1,000-way-10-shot task. It is first initialised using 500 classes from the training set in a standard supervised manner. Once initialised, it learns the remaining 500 classes sequentially, updating only the plastic neural network layers and employing product quantisation to compress the sample embeddings stored in its replay buffer. We then test REMIND on the test set of the task. Like the Offline learner, we repeat this for 20 seeds.

#### Meta-CL Algorithms
Recall that the meta-training phase consists of two loops: the outer loop and the inner loop. A single iteration in the outer loop involves processing a batch of 16 100-way-10-shot episodes. At the start of each episode, we reinitialise the MLP layers for OML, whereas for GeMCL, we initialise the class-specific parameters using the meta-learned priors. The inner loop then involves the agents making use of the support set to learn the classes sequentially. A single inner-loop step involves updating the MLP layers for OML or calculating the class statistics for GeMCL. Once training on the support set is complete, the agent is tested on the query set. We calculate the mean Cross Entropy loss for that episode, average it over the batch, and use it to update the meta-parameters.

After training, we proceed to the meta-test phase where the meta-parameters are fixed. OML and GeMCL are then tested on 512 1,000-way-10-shot episodes using classes they had not seen during training.
The episode procedure mirrors the meta-training phase. At the start of each episode, we reinitialise the MLP layers for OML or initialise the class-specific parameters from the priors for GeMCL, and then use the support set to learn the classes sequentially. Finally, we measure the accuracy on the query set to evaluate performance on these unseen classes.
### Implementation Details
#### Common Settings & Architecture
All algorithms in our experiments make use of the same encoder: a 5-layer CNN with batch normalisation, followed by a linear layer. The input images from both Omniglot and CASIA were resized to $$32\times32$$ tensors. Following Lee et al. <d-cite key="lee_learning_2024"></d-cite>, we applied no data augmentation.

For the classifier head, the Offline learner, REMIND, and OML employ a 2-layer MLP following the encoder with an output shape of 1,000. GeMCL makes use the encoder output directly for its Bayesian updates.

#### Standard Baselines
For the initialisation phases of both the Offline learner and REMIND, we used the `Adam` optimiser <d-cite key="kingma_adam_2017"></d-cite>.

* **Offline Learner:** The model was trained for 65 epochs. To determine the best performance, we selected the checkpoint with the highest accuracy on the test set. We noted that Lee et al. <d-cite key="lee_learning_2024"></d-cite> recorded accuracy when the test *loss* was lowest; however, in our preliminary experiments, we found that the lowest loss did not necessarily yield the highest accuracy.
* **REMIND:** The encoder serves as the frozen layer, while the 2-layer MLP acts as the plastic layers. During the initialisation phase, the model was trained for 100 epochs, and we saved the parameters that achieved the best accuracy on the subset's test set. During the CL stage, we optimised REMIND's plastic layers using SGD with momentum and weight decay. We did not use augmentation during replay; the quantized tensors from the replay buffer were used without modification, unlike in the original work. We configured the PQ model with a codebook size of 128 centroids and 32 subvectors. The replay buffer was set to hold a maximum of 7,000 compressed samples.

#### Meta-Continual Learners
Both OML and GeMCL were trained for 20,000 outer-loop steps, using the `Adam` optimiser for the encoder updates.

* **OML:** We used the second-order derivatives for the meta-update. For the inner loop, we used SGD to update the MLP layers. Instead of performing a hyperparameter search for the inner-loop learning rate, we followed Lee et al.'s <d-cite key="lee_learning_2024"></d-cite> approach of making it a learnable parameter. During meta-testing, we only update the last layer of the MLP, as previous works <d-cite key="beaulieu_learning_2020"></d-cite> have shown this leads to better performance. Javed and White <d-cite key="khurram_javed_meta-learning_2019"></d-cite> also pointed this out in the README of the [official OML repository](https://github.com/kjaved0/mrcl?tab=readme-ov-file).
* **GeMCL:** Although GeMCL can learn classes in any order and so can learn classes in parallel, we performed the learning sequentially in our implementation.

### CASIA Experiment
In this experiment, we compare the meta-learners (OML and GeMCL) to the baselines (REMIND and Offline) using the CASIA dataset. We perform meta-training on the meta-train classes and perform meta-testing on the meta-test classes. The baselines are trained and tested using the meta-test classes.

We compare the average accuracy achieved by the baselines on the test set to the average accuracy achieved by the meta-learners during meta-testing. We report the average accuracy and the 95% confidence intervals, calculated using the bootstrap method.

### Omniglot Experiment
This experiment assesses whether the meta-parameters learned on CASIA can generalise to a different dataset. Here, the meta-learners are only meta-tested on the Omniglot classes; their encoders remain fixed from the CASIA meta-training phase. The baselines are trained from scratch on the Omniglot classes. We compare the average accuracy from the meta-testing phase to the test set accuracy of the baselines. We report the 95% confidence intervals, calculated using the bootstrap method.

## Results and Analysis

### CASIA Results and Analysis
{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/casia_algorithm_accuracy_annotated.png" class="img-fluid" %}
<div class="caption"> Figure 6. The average test accuracy on the CASIA dataset with 95% confidence intervals (CIs). The numerical CI range is annotated above each algorithm. </div> 

Figure 6 displays the average test accuracy on the CASIA dataset for each of the algorithms. OML matches the performance of the Offline learner, whereas GeMCL achieves a higher average accuracy on the 1,000-way-10-shot task. The results indicate that the meta-CL algorithms are able to generalise to tasks with more classes and are able to continually learn 1,000 classes despite only being meta-trained on tasks with a 100 classes. Interestingly, GeMCL outperforms the Offline learner. Lee et al. <d-cite key ="soochan_lee_recasting_2023"></d-cite> noted that it is possible for a meta-learner to surpass an offline learner when the meta-training set is large compared to the training set used by the offline learner. Both meta-CL algorithms outperform the REMIND algorithm as well.

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/casia_comparison_accuracy_over_time.png" class="img-fluid" %}
<div class="caption"> Figure 7. The average test accuracy for the CASIA dataset measured after training was completed. The accuracy is reported separately for each class interval, ordered by the sequence in which the classes were learned (e.g., the first interval represents the first 100 classes encountered during training). The error bars indicate the 95% confidence interval. </div> 

To determine if the algorithms are as accurate on their earliest tasks as they are on their most recent ones, Figure 7 presents the average test accuracy at the end of training, ordered by the sequence in which the classes were learned.

GeMCL demonstrates remarkable consistency, as the plot is virtually a straight line. This supports the theoretical claim that GeMCL does not suffer from catastrophic forgetting. This is expected, given that each class maintains its own statistics and there is no interference because the encoder is fixed.

OML, on the other hand, is less consistent. Its accuracy increases for the later classes, with the lowest accuracy observed for the first 100 classes learned. While it performs well at the end, this trend does hint that perhaps OML suffered from some forgetfulness regarding the earlier tasks. Interestingly, however, on the last 100 classes, it outperforms GeMCL.

REMIND’s accuracy appears consistent across the earlier intervals, likely due to the use of a replay buffer. However, it achieves its lowest accuracy for the last 100 classes learned. It is puzzling that the lowest accuracy occurs on the most recently learned classes, and the reason for this is unclear to us.

### Omniglot Results and Analysis
{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/omniglot_algorithm_accuracy_annotated.png" class="img-fluid" %}

<div class="caption"> Figure 8. The average test accuracy on the Omniglot dataset with 95% confidence intervals (CIs). The numerical CI range is annotated above each algorithm. </div>
Figure 8 displays the average test accuracy on the Omniglot dataset. In this case, the Offline learner performed best. GeMCL did not perform as well as it did on the CASIA dataset, but it still achieved higher accuracy than REMIND on this new dataset on which it was not trained.

{% include figure.liquid path="assets/img/2026-04-27-pushing-meta-cl-methods/omniglot_comparison_accuracy_over_time.png" class="img-fluid" %}
<div class="caption"> Figure 9. The average test accuracy for the Omniglot dataset measured after training was completed. The accuracy is reported separately for each class interval, ordered by the sequence in which the classes were learned (e.g., the first interval represents the first 100 classes encountered during training). The error bars indicate the 95% confidence interval. </div> 

Figure 9 shows the final accuracy against the chronological learning sequence. Figure 9 follows a similar pattern to Figure 7.
GeMCL performs almost identically across all class intervals. OML's accuracy increases as the sequence of classes progresses, with the highest accuracy observed at the end. REMIND again exhibits the same behaviour, where accuracy is lowest for the last 100 classes learned.

Figures 8 and 9 indicate that OML does not generalise as well to a new dataset compared to GeMCL and that some
form of forgetting occurs, as its accuracy increases for the most recently learned classes. REMIND appears largely consistent for the earlier intervals, appearing virtually as a straight line. However, the puzzling behaviour where it achieves the lowest accuracy for the last 100 classes is present here as well.

## Discussion
We found it quite challenging to improve REMIND's performance. We also did not make use of augmentation during replay in our experiments, as we did not see any improvement in performance with it. One hypothesis for improving REMIND's performance is that allowing layers of our neural network to be plastic could help.

Figures 7 and 9 suggest that some forgetting occurs in OML. Kwon et al. <d-cite key="kwon_lifelearner_2024"></d-cite> make use of latent replay to mitigate forgetting in meta-CL, specifically in ANML <d-cite key="beaulieu_learning_2020"></d-cite>, which is another MAML-inspired meta-CL method. This approach achieved high accuracy on audio and image datasets. The replays are compressed using Sparse Bitmap and PQ. They chose ANML over OML because it achieved higher accuracy during evaluation.
However, as mentioned before, if one just fine-tunes the last layer of OML during meta-testing, performance improves. Javed and White <d-cite key="khurram_javed_meta-learning_2019"></d-cite> point out in the OML repo that with this change, they actually match the performance of ANML. In our preliminary experiments, we also found that OML where we only fine-tune the last layer matches ANML. Hence why we decided to just focus on OML. It would then be interesting to apply Kwon et al.'s approach to this version of OML to see how it performs at scale, and compare this approach to GeMCL, REMIND, and the Offline approach.

For OML, we assume that there is a maximum number of classes to be learned, determined by the output nodes of the last layer. Hypersensitivity to the inner-loop learning rate during meta-testing is another concern with the algorithm. However, this hypersensitivity was addressed by Lee et al. <d-cite key="lee_learning_2024"></d-cite> by making the inner-loop learning rate a learnable parameter.

Lee et al. <d-cite key="lee_learning_2024"></d-cite> pointed out that the fact that GeMCL is immune to CF means that one can focus on improving the representation ability of the encoder.
The main focus now with GeMCL is training an encoder such that it can generalise to different datasets and output meaningful representations when applied to different datasets.
As mentioned before, the order of the classes does not matter; one can learn the classes in parallel, which speeds up training. However, Lee et al. <d-cite key="lee_learning_2024"></d-cite> also raised a valid point that if the order of classes matters, then using GeMCL might not be ideal.
Our results show that GeMCL might be able to generalise to different datasets and scale to classification tasks with 1,000 classes. Future work could be to see if one can train a feature extractor that generalises to datasets with different modalities.

It should be noted that the goal of CL is not just to prevent CF. Hadsell et al. <d-cite key="raia_hadsell_embracing_2020"></d-cite> mentioned that a desideratum of CL is forward and backward transfer as well. Forward transfer means that learning previous tasks helps improve learning on future tasks. In other words, the performance on Task $$B$$ after just training on Task $$B$$ would not be as high as the performance on Task $$B$$ after learning Task$$A$$ and $$B$$. Backward transfer refers to learning new tasks improving performance on previously learned tasks. An interesting future work is designing a meta-CL algorithm that has this property.

Another desideratum of CL is that CL algorithms should be more computationally efficient than offline approaches. Future work should go into actually measuring the computational efficiency of meta-CL algorithms compared to the Offline approach and other CL algorithms. As mentioned before, meta-training is expensive and it is often challenging to design the meta-training dataset. It would be interesting to see: even if meta-CL can scale, is it worth doing if the meta-training process is so expensive? But the chance that we could develop a meta-CL algorithm that is able to generalise to any other task, potentially decreasing computational costs, is enticing.

The idea would be to follow the approach of Harun et al. <d-cite key="harun_how_2023"></d-cite> whereby we measure the NetScore metric <d-cite key ="wong_netscore_2019"></d-cite> of the meta-CL algorithms and compare it to CL algorithms that have been shown to be more computationally efficient than the Offline approach, such as REMIND.
The NetScore metric allows us to combine the accuracy, memory, parameters, and compute of a model. The higher the NetScore, the better the model.
Meta-CL algorithms are still CL algorithms, and it is important to see if we obtain any benefit in terms of computational efficiency by performing meta-learning at scale.

## Conclusion
In this work, we aimed to see how well meta-CL algorithms perform on tasks with extended data streams, specifically on 1,000-way-10-shot classification tasks. We found that on the CASIA dataset, OML and GeMCL outperform the CL algorithm REMIND and match the performance of the Offline learner. This suggests that there is a benefit to applying meta-learning to CL, as OML and GeMCL match the performance of the Offline learner.
However, when tested on the Omniglot dataset, after having been trained on the on the CASIA dataset, we observed a drop in performance. Both meta-CL algorithms performed worse than the Offline learner in this case, and OML performed worse than REMIND as well.

Our results also suggest that OML might be suffering from CF. Applying Kwon et al.'s <d-cite key="kwon_lifelearner_2024"></d-cite> approach to OML could help mitigate the CF. We also struggled to get REMIND to perform well. We hypothesised that perhaps making more layers plastic could improve performance of the REMIND algorithm. Having seen that meta-CL can scale to 1,000 classes, future work should focus on how computationally efficient they are and whether they can generalise to new domains.