# Deploying a newly trained model

> Temporary note for SGP-49. The Render part only works once SGP-49 is merged
> and Render is connected.

The deployed app does not use `.pt` checkpoints and does not install torch. It
runs an ONNX model downloaded from the public Hugging Face repo
[`Eclayeser/plant-disease-cnn-demo`](https://huggingface.co/Eclayeser/plant-disease-cnn-demo).
`deployment/model.lock.json` names the exact Hugging Face commit the app uses.

```
train.py ─► model.pt ─► export_onnx ─► deployment/model/ ─► hf upload ─► Hugging Face
                                                                             │
          Render ◄── merge to main ◄── commit model.lock.json ◄── new revision
```

Uploading a model does not deploy it. Only merging a changed
`model.lock.json` to `main` does.

## Before you start

- Activate the venv: `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux.
- Get write access to the Hugging Face repo from its owner, create a token with
  **Write** permission at huggingface.co → Settings → Access Tokens, and log in
  once with `hf auth login`.

## 1. Train

```
python train.py --config configs/<your_config>.yaml --responsible <your_name>
```

This saves a `.pt` checkpoint. You can also download one that is already
trained from ClearML.

## 2. Export to ONNX

```
python -m scripts.export_onnx --checkpoint path/to/model.pt
```

This writes `model.onnx`, `labels.json` and `model_config.json` to
`deployment/model/`, which git ignores. The script stops with an error if the
ONNX model's predictions differ from PyTorch's.

Run it with `python -m`. `python scripts/export_onnx.py` fails with
`ModuleNotFoundError: leafapp`.

## 3. Try it locally

```
python -m app.app
```

Open http://localhost:8080 and upload a few leaf photos. The app loads whatever
is in `deployment/model/`, so you are testing the new model before anyone else
sees it.

## 4. Upload to Hugging Face

If the model is different (architecture, data, image size), update
`deployment/model/README.md` first. It is the model's page on Hugging Face. If
the folder has no README, the one already on Hugging Face stays as it is.

```
hf upload Eclayeser/plant-disease-cnn-demo deployment/model . --commit-message "<what this model is>"
```

Always upload to the same repo. Older versions stay available under their own
commit ids.

## 5. Get the new revision id

```
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Eclayeser/plant-disease-cnn-demo').sha)"
```

Copy the 40-character id it prints.

## 6. Pin it in the lock file

On a new branch, set `revision` in `deployment/model.lock.json` to that id:

```json
{
  "repo_id": "Eclayeser/plant-disease-cnn-demo",
  "revision": "<new 40-character id>"
}
```

## 7. Check the download works

Delete the local folder, so you know the files really come from Hugging Face,
then fetch them again and run the app:

```
python -m scripts.fetch_model_from_hf
python -m app.app
```

To delete the folder, run `Remove-Item -Recurse -Force deployment/model` on
Windows or `rm -rf deployment/model` on macOS/Linux.

## 8. Deploy through `main`

Commit only `deployment/model.lock.json`, open a PR and merge it to `main`.
After the CI checks pass, Render:

1. installs `requirements-deploy.txt` (no torch);
2. runs `python -m scripts.fetch_model_from_hf`, which downloads the pinned revision;
3. starts `python -m app.app`.

The free-tier app sleeps after 15 minutes without visitors. The first visit
after that takes about a minute.

## Rolling back

Set `revision` back to the previous id (see the git history of
`model.lock.json`) and merge to `main`.

## Do not

- Commit `.pt` or `.onnx` files to git.
- Delete or squash commits in the Hugging Face repo. Old lock files point to
  them.
- Change only the UI and the model in the same PR. Keeping them separate makes
  a bad deploy easy to trace.
