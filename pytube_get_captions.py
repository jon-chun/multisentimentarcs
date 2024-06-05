from pytube import YouTube

yt_url = "https://www.youtube.com/watch?v=bnU2dWMGpMA"

yt = YouTube(yt_url)

# NECESSARY HACK
yt.bypass_age_gate()

print(yt.captions)
print(yt.caption_tracks) 


# METHOD #O Working Hack
# https://github.com/pytube/pytube/issues/1794

stream = yt.streams.first() # required to get following portions to work
print(yt.captions)
caption = yt.captions['en']
print(caption.json_captions)



# METHOD #1: Documentation (not working 20240531)
"""
caption = yt.captions.get_by_language_code('en')

caption_xml = caption.xml_captions
print(type(caption_xml))
print(f"CAPTIONS XML:\n\n{caption_xml}\n\n")

caption_srt = caption.generate_srt_captions()
print(type(caption_srt))
print(f"CAPTION SRT: {caption_srt}\n\n")
""";


# METHOD #2: Hack:
# https://github.com/pytube/pytube/issues/1477

# Too Involved: modify source code



