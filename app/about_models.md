<p class="ai-note">ⓘ Predictions come from AI models and can be wrong.</p>

<details class="about-models">
<summary>Read more</summary>

- **Intended use.** We realise many people may treat our models' predictions as a source of truth. However, we recommend using the models only for a first opinion on the likely disease, and we encourage you to confirm the result with an agronomist or plant-health expert before any treatment.
- **Training data.** Our models learned from 55,447 photographs in the PlantVillage dataset, each showing a single leaf on a plain background in a lab. They cover 26 diseases and 12 types of healthy leaf, plus photos without a leaf, so the app can tell you when it can't find one.
- **Crops covered.** Our models can recognise leaves from 14 crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry and tomato. The models haven't been trained on any other plants, so a leaf from another plant will very likely get a wrong prediction.
- **Known limitations.** To get the most reliable result, it helps to know where our models can go wrong:
    - Photos taken outdoors, with busy backgrounds, uneven lighting or several leaves, look different from the photos our models learned from, so results may be less accurate. A clear photo of one leaf on a plain surface works best for our models.
    - The models always pick one of their 39 classes, even for a disease they don't know, so an unfamiliar disease will still get a label, and that label will be wrong.
    - Unfortunately, our training data does not cover every disease for every crop. For blueberry, raspberry, soybean, orange and squash, the models have seen only healthy leaves or only a single disease, so please take some extra time to cross-check the results for these crops.
    - Some leaf types had far fewer training photos than others, so the models are more reliable for some crops than for others.
- **Evaluation.** We tested each model on photos it never saw during training. Please see the results, including how well each crop and disease is recognised, in our [model card](https://github.com/Eclayeser/Plant-Diseases_Image-Classification_CNN/blob/main/documents/model_card.md).

</details>
