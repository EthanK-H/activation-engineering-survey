# Activation Vectors Directory README
This directory contains steering vectors generated using a dataset of true and false statements, tailored to model different personas. Each statement is prefixed with a contextual phrase to guide the model's response style. The general format used is: "Pretend you're {type} person making statements about the world," where {type} varies depending on the desired persona, although the specific wording can vary.

## File Naming Conventions

In this directory, we store the steering vectors used for controlling the outputs of our language models. The file naming convention is designed to clearly indicate the contents and characteristics of each file. Here's the template we follow for naming these files:


### Breakdown of the Naming Convention:

- **Model Name**: The name of the model these steering vectors are applied to.
- **Model Size**: The size of the model, e.g. 7B (7 billion parameters)
- **Sycophancy**: This field is currently a static placeholder, indicating the type of steering vector.
- **Steering Vector Label Number**: A unique identifier for the steering vector. This number differentiates between different sets of steering vectors.
- **Direction or Signs**: Specifies whether the file contains the 'direction' vectors or 'sign' vectors.

## Catalog of Steering Vectors

In this section, you can find details about each steering vector stored in this directory. For each entry, information about the dataset used for generating the steering vector, the specific model it's intended for, and other relevant details are provided.

### Catalog Entries:

1. **Mistral_7b_sycophancy_1**
   - **Model:** Mistral
   - **Model Size:** 7B
   - **Steering Vector Label Number:** 1
   - **Types:**
     - **Directions:** `Mistral_7b_sycophancy_1_directions.pkl`
     - **Signs:** `Mistral_7b_sycophancy_1_signs.pkl`
   - **Description:** Generated using prompts that simulate an 'honest' or 'sycophantic' persona. Honest prompts included "Pretend you're an honest person making statements about the world," and sycophantic prompts used "Pretend you're a sycophantic person making statements about the world."

2. **Mistral_7b_sycophancy_2**
   - **Model:** Mistral
   - **Model Size:** 7B
   - **Steering Vector Label Number:** 2
   - **Types:**
     - **Directions:** `Mistral_7b_sycophancy_2_directions.pkl`
     - **Signs:** `Mistral_7b_sycophancy_2_signs.pkl`
   - **Description:** Generated using prompts that simulate an 'honest' or 'sycophantic' persona. Honest prompts included "Pretend you're someone who is honest, helpful, and always corrects mistakes in a constructive manner, and you are making statements about the world." Sycophantic prompts used "Pretend you're someone who is never honest, you only want to be liked and will never correct the mistakes that people make, and you are making statements about the world."

3. **Mistral_7b_sycophancy_3**
   - **Model:** Mistral
   - **Model Size:** 7B
   - **Steering Vector Label Number:** 3
   - **Types:**
     - **Directions:** `Mistral_7b_sycophancy_3_directions.pkl`
     - **Signs:** `Mistral_7b_sycophancy_3_signs.pkl`
   - **Description:** Generated using prompts that simulate an 'honest' or 'sycophantic' persona. Honest prompts included "Imagine you're a person who values truth and integrity above all, and you're discussing various topics." Sycophantic prompts used "Imagine you're a person who always agrees with others to gain favor, regardless of the truth."

4. **Mistral_7b_sycophancy_4**
   - **Model:** Mistral
   - **Model Size:** 7B
   - **Steering Vector Label Number:** 4
   - **Types:**
     - **Directions:** `Mistral_7b_sycophancy_4_directions.pkl`
     - **Signs:** `Mistral_7b_sycophancy_4_signs.pkl`
   - **Description:** Generated using prompts designed to contrast honesty and sycophancy. Honest prompts used "You are someone who speaks their mind honestly and values factual accuracy, and you're giving your opinion on different matters." Sycophantic prompts included "You are someone who always says what others want to hear, even if it means hiding your true thoughts, and you're giving your opinion on different matters."


## Saving and Loading Steering Vectors

To save and load the steering vectors, we use Python's `pickle` module. Below is the example code demonstrating this process:

### Saving Steering Vectors

```python
import pickle

# Saving directions
with open("path_to_directions_file.pkl", "wb") as f:
    pickle.dump(directions_object, f)

# Saving direction signs
with open("path_to_direction_signs_file.pkl", "wb") as f:
    pickle.dump(direction_signs_object, f)
```

```python
### Loading Steering Vectors
# Loading directions
with open("path_to_directions_file.pkl", "rb") as f:
    directions = pickle.load(f)

# Loading direction signs
with open("path_to_direction_signs_file.pkl", "rb") as f:
    direction_signs = pickle.load(f)
```

## Additional Notes

- Ensure that you understand the structure and purpose of each steering vector before applying it in a model.
- The steering vectors are specific to the model and its size. Do not interchange them between different models or sizes without appropriate adjustments.
- For loading these steering vectors, use Python's pickle module. Be aware of the security implications of using pickle with untrusted data.

