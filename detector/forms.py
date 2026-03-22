"""Forms for the detector app."""

from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(label="Choose an image")


class HoughForm(forms.Form):
    image       = forms.ImageField(label="Image")
    canny_t1    = forms.FloatField(label="Canny threshold 1", initial=50)
    canny_t2    = forms.FloatField(label="Canny threshold 2", initial=150)
    # Lines
    line_thresh = forms.IntegerField(label="Line vote threshold", initial=50)
    min_line_len= forms.FloatField(label="Min line length",    initial=50)
    max_line_gap= forms.FloatField(label="Max line gap",       initial=10)
    # Circles
    dp          = forms.FloatField(label="Circle dp",          initial=1.2)
    min_dist    = forms.FloatField(label="Circle min dist",    initial=30)
    circle_p1   = forms.FloatField(label="Circle param1",     initial=100)
    circle_p2   = forms.FloatField(label="Circle param2",     initial=30)
    min_r       = forms.IntegerField(label="Min radius",       initial=0)
    max_r       = forms.IntegerField(label="Max radius",       initial=0)


class SnakeForm(forms.Form):
    image      = forms.ImageField(label="Image")
    center_x   = forms.IntegerField(label="Contour center X", initial=200)
    center_y   = forms.IntegerField(label="Contour center Y", initial=200)
    radius     = forms.IntegerField(label="Initial radius",   initial=100)
    num_points = forms.IntegerField(label="Contour points",   initial=60)
    alpha      = forms.FloatField(label="α (continuity)",     initial=1.0)
    beta       = forms.FloatField(label="β (curvature)",      initial=1.0)
    gamma      = forms.FloatField(label="γ (image energy)",   initial=1.5)
    window     = forms.IntegerField(label="Search window",    initial=7)
    iterations = forms.IntegerField(label="Max iterations",   initial=100)
