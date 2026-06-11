from django.shortcuts import redirect, render

# Create your views here.

def login_page(request):
    if request.method == "POST":
        return redirect('home')

    return render(request, 'login.html')

def signup_page(request):
    return render(request, 'signup.html')

def home(request):
    return render(request, 'index.html')

def main(request):
    return render(request, 'main.html')

def photo(request):
    return render(request, 'photo.html')

def measurements(request):
    return render(request, 'measurements.html')

def body(request):
    return render(request, 'bodyanalysis.html')

def color(request):
    return render(request, 'color_analysis.html')

def analyse(request):
    return render(request, 'color_analysis.html')

def manual(request):
    return render(request, "manual.html")

def pic(request):
    return render(request, "pic.html")

def profile(request):
    context = {
        "profile_picture": "/static/default-user.png",
        "style_archetype": "Modern Minimalist",

        "total_analyses": 0,

        "color_history": [],
        "body_history": [],
        "saved_outfits": [],

        "fashion_score": {
            "style_consistency": 82,
            "wardrobe_versatility": 76,
            "color_harmony": 88,
            "personal_brand": 79,
        }
    }

    return render(request, "profile.html", context)

def reset(request):
    return render(request, "reset.html")