from datasets import load_dataset
dataset = load_dataset('AiresPucrs/CelebA-Smiles', split="train")

def resize_image(examples):
    examples["resized_image"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
    return examples

# Apply `resize_image` funtion to the dataset
dataset = dataset.map(resize_image, batched=True)
path = "../data/celeba/"
for i in range(10):
    dataset[i]['resized_image'].save(f"{path}{i}.png")
