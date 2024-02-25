import face_recognition
import cv2
import os

# Ruta al directorio que contiene las imágenes
directorio_imagenes = "/ruta/a/tu/directorio"

# Obtener la lista de nombres de archivos en el directorio
nombres_archivos = os.listdir(directorio_imagenes)

# Cargar las imágenes y codificaciones faciales
codificaciones_conocidas = []
nombres_conocidos = []

for nombre_archivo in nombres_archivos:
    ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
    
    # Ignorar archivos que no sean imágenes
    if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Cargar la imagen y obtener la codificación facial
    imagen = face_recognition.load_image_file(ruta_imagen)
    codificacion = face_recognition.face_encodings(imagen)[0]
    
    # Agregar la codificación y el nombre
    codificaciones_conocidas.append(codificacion)
    nombres_conocidos.append(os.path.splitext(nombre_archivo)[0])  # Quitar la extensión del archivo

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar el frame de la cámara
    ret, frame = cap.read()

    # Encontrar todas las caras en el frame
    caras_en_frame = face_recognition.face_locations(frame)
    codificaciones_en_frame = face_recognition.face_encodings(frame, caras_en_frame)

    # Iterar sobre las caras en el frame
    for (top, right, bottom, left), codificacion_en_frame in zip(caras_en_frame, codificaciones_en_frame):
        # Comparar la codificación facial con las codificaciones conocidas
        coincidencias = face_recognition.compare_faces(codificaciones_conocidas, codificacion_en_frame)

        nombre = "Desconocido"

        # Si hay una coincidencia, obtener el nombre correspondiente
        if True in coincidencias:
            indice_coincidencia = coincidencias.index(True)
            nombre = nombres_conocidos[indice_coincidencia]

        # Dibujar un rectángulo alrededor de la cara y mostrar el nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nombre, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Mostrar el frame resultante
    cv2.imshow('Reconocimiento Facial', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
