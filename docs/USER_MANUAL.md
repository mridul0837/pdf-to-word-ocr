# PDF to Word OCR — User Manual

This app turns PDFs, scans and photos of documents (even handwriting) into Word
documents you can edit.

**Everything happens on your computer. Your files are never uploaded, and no internet
is needed.**

## What you need

- A Windows 10 or Windows 11 computer.
- About 3.5 GB of free disk space while setting up (the download is 1.1 GB and the
  unpacked app is 2.2 GB). You can delete the download afterwards.

There is nothing else to install.

## First time: set it up (about 2 minutes)

1. Download **PDFtoWordOCR-portable.zip**. If Google Drive says it can't scan the file
   for viruses, click **Download anyway**.
2. Right-click the ZIP, choose **Extract All…**, then click **Extract**.
3. Open the new **PDFtoWordOCR** folder and double-click **PDFtoWordOCR.exe**.
4. If a blue box says **"Windows protected your PC"**, click **More info**, then
   **Run anyway**. It appears because the app doesn't come from the Microsoft Store.

> **Desktop icon:** right-click **PDFtoWordOCR.exe**, then choose **Send to → Desktop
> (create shortcut)**. On Windows 11, click **Show more options** first.
> Keep the .exe inside its folder, because it needs the files next to it.

## Convert a document

1. Click **Choose file…** and pick your PDF or picture (JPG, PNG and so on). You can
   also drag the file onto the window.
2. Choose the **Mode** (see the table below).
3. Click the blue **Extract text** button. Wait until the line under the progress bar
   says **Finished**.
4. Read the text in the big box and fix any mistakes. You can type in it, just like in
   Notepad.
5. Click **Save Word document**. By default the Word file is saved next to your original
   file, with `_ocr` added to its name. To save it somewhere else, click **Browse…** next
   to **Save Word file as** first.

## Which mode should I pick?

| Your document | Mode |
|---|---|
| A normal PDF made on a computer (you can select its text) | **Auto** |
| A scan or photo of a typed or printed page, or a receipt | **Auto** or **Printed Text** |
| Handwriting | **Handwriting** |

Auto never switches to Handwriting by itself. If a handwritten page comes out as
gibberish, choose **Handwriting** yourself.

Handwriting mode is slower, about half a minute per page, and the first page takes a
little longer. It reads English handwriting.

## Tips for good results

- Use clear, straight, well-lit scans or photos.
- **Receipts:** lay the receipt flat, fill the photo with it, and avoid shadows.
  Crumpled or folded receipts still read poorly.
- Always double-check names, numbers and dates.
- Keep **Join lines into paragraphs** ticked for normal text. Untick it for poems,
  addresses or lists, to keep the original line breaks.
- Lines like `===== PAGE 2 · … =====` show where a new page starts. They become page
  breaks in Word. You can delete them if you don't want page breaks.
- **Language** and **DPI** can stay as they are. A higher DPI reads small print better
  but is slower.

## If something goes wrong

| What you see | What to do |
|---|---|
| The text is gibberish | The page is probably handwritten. Choose **Handwriting** mode and click **Extract text** again. |
| A **"Low OCR confidence"** message | Same as above. If it isn't handwriting, try a clearer scan. |
| **"… lines were too unclear to read and left out"** | Handwriting mode skipped lines it couldn't read, rather than guess. Type those lines in from the original. |
| It seems stuck | Wait: handwriting takes a while. The progress bar and the status line show what it's doing. **Cancel** stops it. |
| **"Could not write … open in Word"** | Close that document in Word, then click **Save Word document** again. |
| **"not a valid PDF or image file"** | The file is damaged, or it isn't a PDF or picture. Check that it opens in another program. |
| **"Windows protected your PC"** | Click **More info**, then **Run anyway**. |

## Privacy

Your documents never leave your computer, and the app works without internet.

## Removing the app

Delete the **PDFtoWordOCR** folder, and the desktop shortcut if you made one.
