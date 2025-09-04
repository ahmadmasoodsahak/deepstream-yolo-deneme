import pyds
import gi

gi.require_version('Gst', '1.0')
import gi.repository.Gst as Gst
import gi.repository.GLib as GLib

import pandas as pd
import numpy as np

Gst.init(None)

def create_pipeline(video_source):
    # Pipeline oluşturma
    pipeline = Gst.Pipeline.new("test-pipeline")
    
    # Video kaynağını oluşturma
    source = Gst.ElementFactory.make("filesrc", "source")
    source.set_property("location", video_source)

    # Video decoder ekleme
    decoder = Gst.ElementFactory.make("decodebin", "decoder")
    
    # Video converter ekleme
    converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
    
    # Caps filter - NVMM format için
    caps_nvmm = Gst.ElementFactory.make("capsfilter", "caps_nvmm")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM)")
    caps_nvmm.set_property("caps", caps)
    
    # Streammux - DeepStream batch meta oluşturmak için gerekli
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    # Nvinfer (inference engine) oluşturma
    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", "../DeepStream-Yolo/config_infer_primary_yoloV8.txt")
    
    # Tracker ekleme - konfigürasyon dosyası ile
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    
    # osd nesneleri ekleme
    nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
    
    # Video converter (sink için)
    converter2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")
    
    # Caps filter (sink için)
    caps_filter = Gst.ElementFactory.make("capsfilter", "caps_filter")
    caps_sink = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    caps_filter.set_property("caps", caps_sink)
    
    # ekranda gösterme elemanı oluşturma
    sink = Gst.ElementFactory.make("nveglglessink", "sink")
    
    # pipeline elemanlarını ekleme
    pipeline.add(source)
    pipeline.add(decoder)
    pipeline.add(converter)
    pipeline.add(caps_nvmm)
    pipeline.add(streammux)
    pipeline.add(nvinfer)
    pipeline.add(tracker)
    pipeline.add(nvosd)
    pipeline.add(converter2)
    pipeline.add(caps_filter)
    pipeline.add(sink)
    
    # elemanları birbirine bağlama
    source.link(decoder)
    decoder.connect("pad-added", on_decoder_src_pad, converter)
    converter.link(caps_nvmm)
    
    # Streammux sink pad'i manuel olarak al ve bağla
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_nvmm.get_static_pad("src")
    srcpad.link(sinkpad)
    
    streammux.link(nvinfer)
    nvinfer.link(tracker)
    tracker.link(nvosd)
    nvosd.link(converter2)
    converter2.link(caps_filter)
    caps_filter.link(sink)
    
    # OSD sink pad'e probe ekleme
    osd_sink_pad = nvosd.get_static_pad("sink")
    if osd_sink_pad:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    return pipeline

def on_decoder_src_pad(decoder, pad, next_element):
    # Converter'a bağla
    sinkpad = next_element.get_static_pad("sink")
    if sinkpad and not sinkpad.is_linked():
        pad.link(sinkpad)

import pyds

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    
    try:
        # DeepStream versiyonuna göre farklı fonksiyon adları denenir
        batch_meta = None
        
        # Farklı PyDS fonksiyon adlarını dene
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        except AttributeError:
            try:
                batch_meta = pyds.get_nvds_batch_meta(hash(gst_buffer))
            except AttributeError:
                try:
                    batch_meta = pyds.glist_get_nvds_batch_meta(gst_buffer)
                except AttributeError:
                    # Son deneme - direkt buffer'dan meta al
                    batch_meta = pyds.NvDsBatchMeta.cast(pyds.gst_buffer_get_nvds_batch_meta_list(gst_buffer).data)
        
        if not batch_meta:
            return Gst.PadProbeReturn.OK
            
        l_frame = batch_meta.frame_meta_list
        detections = []
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                l_obj = frame_meta.obj_meta_list
                
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        r = obj_meta.rect_params
                        detections.append({
                            "class_id": obj_meta.class_id,
                            "confidence": obj_meta.confidence,
                            "bbox": [r.left, r.top, r.width, r.height]
                        })
                        
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                        
                l_frame = l_frame.next
            except StopIteration:
                break
        
        if detections:
            save_detections_to_csv(detections)
            print(f"Detected {len(detections)} objects")
            
    except Exception as e:
        print(f"Error in probe: {e}")
    
    return Gst.PadProbeReturn.OK

def save_detections_to_csv(detections, filename="detections.csv"):
    df = pd.DataFrame(detections)
    df.to_csv(filename, mode='a', header=False, index=False)

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, Debug: {debug}")
        loop.quit()
    return True

def print_performance_data():
    print("Performance data...")
    return True

def main():
    loop = GLib.MainLoop()
    
    video_source = "../videolar/raw_video.mp4"
    pipeline = create_pipeline(video_source)
    
    # Bus mesajlarını dinleme
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Videoyu oynatma
    pipeline.set_state(Gst.State.PLAYING)
    
    # performans verisi
    GLib.timeout_add(5000, print_performance_data)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()