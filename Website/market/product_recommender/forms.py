from django import forms

class ProductSearchForm(forms.Form):
    query = forms.CharField(label='Search Product', max_length=100)
