from django import forms

class ProductSearchForm(forms.Form):
    query = forms.CharField(
        label='Search Product',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Search product by name, brand, commodity...',
            'class': 'form-control'
        })
    )
