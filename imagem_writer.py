from PIL import Image, ImageDraw

# Abre a imagem
image = Image.open("imagem.jpg")

# Cria um objeto ImageDraw
draw = ImageDraw.Draw(image)

# Desenha um retângulo vermelho
draw.rectangle([(50, 50), (200, 200)], fill="red", outline="black")

# Desenha um círculo azul
draw.ellipse([(250, 250), (400, 400)], fill="blue", outline="black")

# Desenha uma linha verde
draw.line([(450, 450), (600, 600)], fill="green", width=5)

# Salva a imagem modificada
image.save("imagem_modificada.jpg")