# MLP Group Project

### Meeting Minutes (2/2/18) & Agenda for Monday (5/2/18)

#### 1. Document Outline

Within this document, a record of our group (g74) meeting on Friday the 2<sup>nd</sup> of February is provided. Initially, each discussed project proposal is outlined with its associated backup plan (§2, §3). In addition, a discussion surrounding the choice of dataset is presented (§4). Thereafter, a set of questions for our tutor are offered (§5). Finally, a preliminary summary of each section contained within the interim is documented (§6).

#### 2. Project Proposal One
* Take a large, labelled dataset and train a fairly basic CNN architecture to obtain some baseline classification benchmarks.
* Subsample the original dataset to obtain either a single much smaller dataset or multiple smaller datasets: **S = {s<sub>1</sub> → s<sub>n</sub>}**.
* Apply suggested techniques **{t<sub>1</sub> → t<sub>n</sub>}** onto **S** in order to obtain modified versions of **S**, such that the new instances **S<sub>transformed</sub>** can be used on alternative architectures and benchmarks can be compared to the originals.

**2.1 Suggested Techniques:**

1. Deep Features
2. One Shot Learning
3. Transfer Learning
4. Data Augmentation

**2.2. Backup Plan:**

* Instead of training novel architectures using the smaller datasets in S and comparing the results to the original benchmarks, we could apply some techniques from **{t<sub>1</sub> → t<sub>n</sub>}** to the original dataset, use the same architecture and compare the results to the original results.

#### 3. Project Proposal Two

Essentially the same project as above, but instead of using all four suggested techniques on the subsampled dataset(s), only transfer learning and one-shot learning would be concentrated on and the proposed backup plan would be replaced by the use of data augmentation.

#### 4. Data

Faces DB VS clothes: http://cvit.iiit.ac.in/projects/IMFDB/

#### 5. Questions for our Tutor

1. Assuming that our project proposals are valid, which methodology would be more appropriate for obtaining our initial baseline benchmarks: **a complex architecture with a dataset that has subtle differences between the classes** or **a basic architecture with a dataset with apparent differences between the classes?** To further elaborate, our intuition indicates that if a dataset with subtle differences between classes were to be employed, a network with a larger depth would be required. Whereas, a dataset with obvious differences between classes would not need to be as sophisticated.
2. Would simplifying our classification problem to be a binary classification task, rather than a multi-class classification task be appropriate? Does this depend on the answer to the previous question? What approach would you suggest?
3. What exactly can be considered to be a backup plan? Are our backup plans strong enough do you think?
4. What level of sophistication of experiment is expected to be reported on within the interim report? Would a set of baseline experiments using a basic CNN on a large dataset (assuming our proposed projects are valid) be ample for a decent grade, or would you suggest further experimentation?
5. Are there any other alternative approaches or experiments you think would be appropriate based on our discussion thus far?
6. We understand that we are suppose to share some documents with you, our tutor. Do you also require access to our Trello board, github repo and Slack channel? As some notes, minutes, etc. will be stored in places other than Google docs.

#### 6. The Interim Report Summary

Within this section each portion of the interim report is discussed in light detail.

**6.1. Motivation and Introduction**

An investigation into viable techniques to be employed in order to boost performance of smaller datasets within the domain of neural architectures is the primary motivation behind the aforementioned project proposals.

**6.2. Aims and Objectives**

To try and find a good method of improving performance on the same task using a smaller version of the same dataset in combination with techniques **{t<sub>1</sub> → t<sub>n</sub>}**.

**6.3. Data Set and Task**

* Faces dataset (http://cvit.iiit.ac.in/projects/IMFDB/)
* Clothes dataset - not sure of the URL.
* Classification task (binary or multi-class).

**6.4. Research Questions**

* How can we improve the performance of small datasets on neural architectures?
* How does data augmentation affect the generalisation of the original model on the original dataset.
* Is data augmentation a valid technique for dealing with subsamples of the same dataset.
* What kind of techniques can we employ to achieve a 'decent' level of accuracy using a subsample of the same dataset - Transfer Learning, Data Augmentation, One Shot Learning, Deep Feature Extraction.

**6.5. Experiments Phase One**

Obtaining benchmarks using a proposed dataset (§6.3) with a basic CNN architecture.

**6.6. Remaining Project Plan**

* Take subsamples of a proposed dataset (§6.3).
* Employ techniques **{t<sub>1</sub> → t<sub>n</sub>}** (§2.1) on subsamples **{s<sub>1</sub> → s<sub>n</sub>}** (§2) and review performance.
* Produce a formal description (academic paper) documenting our findings.