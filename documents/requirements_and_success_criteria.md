# Requirements and success criteria

This document specifies what our plant disease classifier is required to achieve and the criteria by which we judge whether it has done so. The system is intended for people who notice that a leaf looks unhealthy and want a first opinion on the likely cause. Given a photo, it returns the five most probable of 39 classes, which cover 26 diseases, 12 types of healthy leaf and photos without a leaf across 14 crops. Because the answer may influence how a plant is treated, our objective is a first opinion that is correct in most cases, presents its alternatives and states clearly when it should not be relied on.

Each requirement is paired with a success criterion, which is the observable result showing that the requirement has been met, and with the metric or test used to verify it. All criteria were fixed before any final model was evaluated on the test set, so that the thresholds could not be adjusted to the results.

## Stakeholders and their needs

The requirements follow from the needs of the three groups that the system affects, which are its stakeholders. Users, such as growers and gardeners, need a plausible answer within seconds, expect the correct class to be among the options shown, and should be told what the tool cannot do and what happens to their photos. The module team, acting as our client, requires a CNN built from scratch, an interface showing the top five predictions, a public deployment and tracked experiments, each supported by evidence. Our own team maintains the code and therefore needs every reported result to be reproducible and every change to be checked automatically.

## Evaluation protocol

The model criteria (F1 to F4, N1 and N2) apply to each of the three models published for the app: the Wider+Deeper CNN (seed 45) and the Wider CNN (seed 44) from the final production runs (SGP-81), and ResNet18 (seed 44) from the transfer-learning runs (SGP-69). Each model is evaluated once with `evaluate.py` on the held-out test set of its own seed, which comprises 15% of the PlantVillage data (about 8,300 photos) with the same class proportions as the full data set. We refer to this step as the test-set evaluation (SGP-74).

The table below gives the validation results of the three models, which were the basis for selecting them. Our two CNNs were trained from scratch, whereas ResNet18 started from ImageNet weights and serves as the transfer-learning comparison.

| Model | Top-1 accuracy | Top-5 accuracy | Macro recall |
|---|---|---|---|
| Wider+Deeper CNN (seed 45) | 97.22% | 99.90% | 95.8% |
| Wider CNN (seed 44) | 96.74% | 99.92% | 95.1% |
| ResNet18, transfer learning (seed 44) | 99.09% | 99.99% | 98.7% |

## Functional requirements

| ID | Requirement | Success criterion | How it is checked | Result |
|---|---|---|---|---|
| F1 | The model identifies the class of a leaf photo among 39 classes. | Top-1 accuracy on the test set is at least 95%. | Test-set evaluation | |
| F2 | Performance on unseen photos matches the validation performance on which our design decisions were based. | Top-1 accuracy on the test set is at most 1.5 percentage points below the validation accuracy of the same model. | Test-set evaluation compared with the validation results | |
| F3 | The app shows the five most probable classes with their probabilities. | The correct class is among the five for at least 99% of test photos. | Top-5 accuracy in the test-set evaluation; `test_predict_returns_top_5_html` | |
| F4 | The app tells the user when a photo contains no leaf. | At least 90% of test photos without a leaf are recognised as such, in which case the app shows "No leaf found". | Recall of `Background_without_leaves` in the test-set evaluation; `test_predict_warns_when_no_leaf_found` | |
| F5 | The deployed model is the model that was evaluated. | The app applies the same preprocessing as the evaluation, and the exported ONNX model reproduces the PyTorch outputs within 1e-4. | `test_preprocess_image_matches_eval_transform`; parity check in `scripts/export_onnx.py`; `deploy-check` workflow | |
| F6 | The app is publicly accessible. | The app is reachable at a public URL without a login, and `deploy-check` passes on `main`. | `deploy-check` on every push; URL in the README | |
| F7 | The app supports example images and batch prediction. | Selecting a gallery example returns a prediction, and a batch of up to 20 photos returns one result per photo. | `test_predict_works_on_example_images`; `test_predict_batch_returns_one_row_per_image` | |

## Non-functional requirements

| ID | Quality | Requirement | Success criterion | How it is checked | Result |
|---|---|---|---|---|---|
| N1 | Fairness | Rare classes are recognised as reliably as common ones. | Macro-averaged recall is at least 90%, and every class with a test recall below 90% is named in the model card. | Per-class results of the test-set evaluation | |
| N2 | Safety | A "healthy" answer can be relied on. | The mean precision of the 12 healthy classes is at least 90%. | Per-class results of the test-set evaluation | |
| N3 | Robustness | Input that cannot be processed is rejected with a clear message while the app keeps running. | This holds for an unreadable file, a file above 10 MB, an image above 13 megapixels and a batch of more than 20 files. | Invalid-input tests in `tests/test_app.py`, such as `test_unreadable_file_shows_a_clear_error` and `test_both_tabs_reject_too_large_images` | |
| N4 | Efficiency | Predictions are returned quickly within the 512 MB of memory of Render's free instance. | A prediction appears within 10 seconds of submitting a phone photo once the service is awake, and the app does not restart for the largest accepted image or a full batch. | Manual session on the deployed app: 10 timed predictions, reporting the median and the maximum | |
| N5 | Privacy | Uploaded photos are not retained. | No upload remains on the server for more than 20 minutes, and flagging and usage analytics are disabled. | `delete_cache` set to `(600, 600)` in `app/app.py`; `render.yaml`; `test_build_ui_deletes_old_uploads` | |
| N6 | Transparency | Users can see what the tool is intended for and where it fails before relying on it. | Both tabs state that the result is not a diagnosis, and the "Read more" panel describes the intended use, the 14 crops and the known limitations and links to the model card. | `test_description_says_not_a_diagnosis`; `test_about_panel_explains_the_models` | |
| N7 | Reproducibility | Every reported result can be reproduced. | Each published model has a committed configuration and seed, a ClearML training task and a model ID in the README, and all tests pass on `main` at submission. | `configs/`; ClearML task links in the experiment write-ups; `tests.yml` | |

## Justification of the thresholds

The thresholds express the minimum that the tool needs in order to be useful. Validation results were used only to confirm that these minima are attainable.

The first label is the answer that most users will act on, so F1 requires it to be correct for at least 95% of test photos. We consider this the level at which the first label can be trusted as a first opinion, and it lies far above trivial baselines, since random guessing yields 2.6% (1 in 39) and always predicting the largest class yields 9.9%. Top-5 accuracy (F3) is held to a stricter 99%, because the interface depends on the correct class being among the five shown, so that a wrong first label can still be corrected by the user. This tolerates at most 83 failures in about 8,300 test photos, against 8 or fewer on validation.

A high accuracy does not guarantee that the model works well in every respect, so we added two further criteria. The first concerns rare classes. The classes differ greatly in size, from 152 photos (Potato healthy) to about 5,500 (Orange Huanglongbing), so accuracy mostly reflects the large classes. If the model misclassified all 23 test photos of Potato healthy, for example, accuracy would fall by only 0.3 points. Macro-averaged recall (N1) gives every class the same weight and therefore shows such a failure. The second concerns the most costly error for a user, which is a diseased leaf labelled as healthy, since the plant then goes untreated. The precision of the healthy classes (N2) measures how often a "healthy" answer is in fact correct.

For these two criteria and for the recognition of photos without a leaf (F4) the minimum is 90%. They are computed on much smaller groups of photos than the full test set, so their results vary more from one split to another and a lower minimum is appropriate. For the same reason no threshold is set for individual classes. With only 23 test photos of Potato healthy, a single error changes the recall of that class by 4.3 points. Per-class recall is therefore reported in full, and weak classes are named in the model card (N1).

All thresholds are minimum levels and lie below the validation results of all three models. F2 checks separately that those validation results hold on unseen photos, by requiring test accuracy to lie within 1.5 percentage points of validation accuracy. A small difference is expected, because the two sets contain different photos and the final models were selected on the validation set. An accuracy of about 97% measured on 8,300 photos has a standard error of about 0.2 percentage points, so chance alone can account for a difference of about 0.5 points, and the tolerance of 1.5 points leaves room for both effects.

The remaining limits follow from how the system is used and built. Ten seconds (N4) is the limit usually given in usability research for keeping a user's attention on a task (Nielsen, 1993), and it leaves room for uploading a phone photo to the free instance. The 20 minutes in N5 results from the upload clean-up, which runs every 10 minutes and deletes files older than 10 minutes, so a photo uploaded just after one run survives the next run and is removed by the one after.

## Results reported without a threshold

Two results are reported without a pass mark. The first is robustness to degraded photos, for which the test-set evaluation reports accuracy under each corruption alongside accuracy on clean photos. As this has not been measured before, any threshold set now would be arbitrary. We expect perturbations within the range of our training augmentation (flips, rotations of up to 15 degrees, and brightness and saturation changes of up to 20%) to cost little accuracy, whereas blur and noise were absent from training and may cost more.

The second is the comparison between our CNNs and ResNet18, reported with the same metrics. The brief treats this comparison as optional and sets no target. Two points matter for its interpretation. The CNNs have 1.0 and 1.4 million parameters against 11.2 million for ResNet18, and the Wider+Deeper CNN is tested on a different split (seed 45) from the other two models (seed 44), so differences below about 0.5 points may be due to chance.

## Limitations

Meeting these criteria shows that the models perform well on photos like those in PlantVillage, in which a single leaf is photographed against a plain background under controlled lighting. It does not establish how they perform in the field. Mohanty et al. (2016) reported 99.35% accuracy on held-out PlantVillage photos but only about 31% on photos from other sources, and Noyan (2022) reached 49.0% using only 8 background pixels, against 2.6% for random guessing, which indicates that part of any PlantVillage score reflects how the photos were taken. Field use is therefore outside the scope of these criteria. The "Read more" panel in the app warns users that photos taken outdoors may be classified less accurately, and the model card discusses this in more detail.

## Results

The Result columns will be completed after the test-set evaluation with the measured value, whether the criterion was met, and a link to the corresponding ClearML task. A criterion that is missed will remain in the table as missed, together with its most likely cause.

## References

- Mohanty, S. P., Hughes, D. P. and Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419. doi:10.3389/fpls.2016.01419
- Nielsen, J. (1993). Response times: the 3 important limits. Excerpt from Chapter 5 of Usability Engineering. https://www.nngroup.com/articles/response-times-3-important-limits/
- Noyan, M. A. (2022). Uncovering bias in the PlantVillage dataset. arXiv:2206.04374
