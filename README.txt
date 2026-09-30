CSI 471/571 Homework 2 - Anirudh Amuda

Included: report, Python source, input images and camera video, all output images,
float64 filter arrays, Haar coefficients and reconstruction, 120 original PNG
frames, 119 difference PNG frames, and four AVI videos.

Reproduce with Python 3.10+ and FFmpeg on PATH:
  python -m pip install -r requirements.txt
  python homework2.py --video data/input.mp4
  python make_report.py

Submission archive: Amuda_A_Homework2.zip.
Read Homework2_Report.pdf and confirm your identity and human collaborator
information before submitting. Human collaborators: none reported.
AI assistance is disclosed in the report.

Input video is preserved as supplied. Its first five seconds are processed at
24 fps in a 320 x 240 canvas, with aspect ratio and orientation preserved.
Camera motion is slight but visible; this limitation is described in the report.

Haar maximum reconstruction error: 0.0.
Both FFV1 outputs preserve their processed RGB frames exactly.
Image provenance: lenag.png was extracted from the supplied assignment DOC.
Synthetic testing clips are not included as assignment data.
