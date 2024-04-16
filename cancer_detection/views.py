from django.shortcuts import render
from .models import UploadedImage
from .predict import predict_histo_vs_rando, predict_histopathological, predict_mammo_vs_rando, predict_mammographical


# Create your views here.
def predict_disease(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        uploaded_image = UploadedImage.objects.create(image=image)
        image_path = uploaded_image.image.path
        # Check if the image is histopathological
        histo_vs_rando_result = predict_histo_vs_rando(image_path)
        if histo_vs_rando_result == 'Histopathological Image':
            # Predict if the image has cancer or not
            histopathological_result = predict_histopathological(image_path)
            result = histopathological_result
        else:
            # Check if the image is mammographical
            rando_vs_mammo_result = predict_mammo_vs_rando(image_path)
            if rando_vs_mammo_result == 'Mammographical Image':
                # Predict if the mammographical image has cancer or not
                mammographical_result = predict_mammographical(image_path)
                result = mammographical_result
            else:
                result = rando_vs_mammo_result
        return render(request, 'result.html', {'result': result})
    return render(request, 'index.html')
