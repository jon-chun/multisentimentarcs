import ollama


image_list = [
    "../data/keyframes/frame0001.png",
    "../data/keyframes/frame0002.png",
    "../data/keyframes/frame0003.png",
    "../data/keyframes/frame0004.png",
    ]

def describe_image(image_path):

    res = ollama.chat(
        model="llava-llama3",
        messages=[
            {
                'role': 'user',
                'content': 'Describe this movie still and the sentiment or emotions evoked by film-maker elements including Facial Expression, Camera Angle, Lighting, Framing and Composition, Setting and Background, Color, Body Language and Gestures, Props and Costumes, Depth of Field, Character Positioning and Interaction, Visual Effects and Post-Processing.',
                'images': [
                    image_path
                    ]
            }
        ]
    )

    image_description = res['message']['content']

    return image_description

for image_index, image_path in enumerate(image_list):
    image_description = describe_image(image_path)
    print(f"Image #{image_index}:\n\n{image_description}\n\n")