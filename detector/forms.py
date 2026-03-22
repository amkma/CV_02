"""Forms for the detector app."""

from django import forms


def _range(mn, mx, step, initial):
    """Return a NumberInput widget with range-slider attrs."""
    return forms.NumberInput(attrs={
        "min": mn, "max": mx, "step": step,
        "value": initial, "data-slider": "true",
    })


class HoughForm(forms.Form):
    image       = forms.ImageField(label="Image")
    canny_t1    = forms.FloatField(label="Canny threshold 1", initial=50,   widget=_range(0, 500, 1, 50))
    canny_t2    = forms.FloatField(label="Canny threshold 2", initial=150,  widget=_range(0, 500, 1, 150))
    line_thresh = forms.IntegerField(label="Line vote threshold", initial=50, widget=_range(1, 200, 1, 50))
    min_line_len= forms.FloatField(label="Min line length",   initial=50,   widget=_range(1, 500, 1, 50))
    max_line_gap= forms.FloatField(label="Max line gap",      initial=10,   widget=_range(1, 100, 1, 10))
    dp          = forms.FloatField(label="Circle dp",         initial=1.2,  widget=_range(0.1, 5.0, 0.1, 1.2))
    min_dist    = forms.FloatField(label="Circle min dist",   initial=30,   widget=_range(1, 200, 1, 30))
    circle_p1   = forms.FloatField(label="Circle param1",    initial=100,  widget=_range(1, 500, 1, 100))
    circle_p2   = forms.FloatField(label="Circle param2",    initial=30,   widget=_range(1, 200, 1, 30))
    min_r       = forms.IntegerField(label="Min radius",      initial=0,    widget=_range(0, 500, 1, 0))
    max_r       = forms.IntegerField(label="Max radius",      initial=0,    widget=_range(0, 500, 1, 0))


class SnakeForm(forms.Form):
    image      = forms.ImageField(label="Image")
    center_x   = forms.IntegerField(label="Contour center X", initial=200,  widget=_range(0, 2000, 1, 200))
    center_y   = forms.IntegerField(label="Contour center Y", initial=200,  widget=_range(0, 2000, 1, 200))
    radius     = forms.IntegerField(label="Initial radius",   initial=100,  widget=_range(1, 500, 1, 100))
    num_points = forms.IntegerField(label="Contour points",   initial=60,   widget=_range(8, 200, 1, 60))
    alpha      = forms.FloatField(label="α (continuity)",     initial=1.0,  widget=_range(0, 10, 0.1, 1.0))
    beta       = forms.FloatField(label="β (curvature)",      initial=1.0,  widget=_range(0, 10, 0.1, 1.0))
    gamma      = forms.FloatField(label="γ (image energy)",   initial=1.5,  widget=_range(0, 10, 0.1, 1.5))
    window     = forms.IntegerField(label="Search window",    initial=7,    widget=_range(3, 21, 2, 7))
    iterations = forms.IntegerField(label="Max iterations",   initial=100,  widget=_range(1, 500, 1, 100))
