# KeySIUserInstruction

This document provides a simple guide to using KeySI.

---

### Before You Start
Make sure all required packages listed in the **requirements.txt** file are installed. You can also change the output directory in **KEYSI_UI.py** by modifying the path variable `OUTPUT_DIR = "KeySI_results"` if you want to save results to another folder.

---

# Main workflow

## 1) Choose the number of groups

In the top-left panel, select how many concept groups you want to create.

## 2) Generate groups

Click **Generate**. New groups (`Group1`, `Group2`, …) will appear in the lower-left panel.

## 3) Assign keywords to groups

- Click a group name (e.g., `Group1`) to make it active.
- In the 2D keyword plot (top-left), select keywords to add to the active group.

## 4) Use Exclude for out-of-scope keywords

If a keyword is irrelevant or consistently introduces noise, add it to **Exclude** so it does not contribute to training.

## 5) Train

Click **Train**. The UI will show **Training...**. When training finishes, a new page will open: **Training View**.

---

# Training View

**Training View** shows document positions before and after training. Use it to check whether the model update matches your intent.

If you are not satisfied:

- go back to the main UI,
- adjust keyword assignments (or group definitions),
- and train again.

---

# Refinement View

After training, open **Refinement View** to inspect and correct the documents used during training.

You can:

- move a document between groups (e.g., `Group1 → Group2`)
- move a document to **Exclude** if it does not belong to any target group
- click points in the 2D view to inspect nearby documents and decide whether reassignment is needed

After finishing adjustments, click **Refinement** again to generate an updated result. Repeat as needed.

---

# Summary

1. Set group count → **Generate**
2. Assign keywords (+ optionally **Exclude**)
3. **Train** → check **Training View**
4. Use **Refinement View** for document-level correction → rerun when needed
