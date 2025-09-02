PRO ADD_PROJECTION
   root_path='/data2/nepal_national_30year/origin_image/Landsat/'
  
  year_start=2018;
  data_scale_str='79E27N 83E26N 85E26N'
  data_scale=strsplit(data_scale_str,/EXTRACT)
  FOR scale_index=0,2 DO BEGIN
    FOR year_index=0,0 DO BEGIN
      year=year_start+year_index
      year_next=year_start+(year_index+2);1
      print,year_next
      year_str=STRTRIM(string(year),1)
      year_next_str=STRTRIM(string(year_next),1)
      result_png_path='/data2/nepal_national_30year/origin_image/'+data_scale[scale_index]+year_next_str+"-result_classification_clean_0808_0729.png"
      tif_path='/data2/nepal_national_30year/origin_image/'+data_scale[scale_index]+year_next_str+"-01-01comp_snow.tif"
      result_tiff_path='/data2/nepal_national_30year/origin_image/product/'+data_scale[scale_index]+year_next_str+"-result_classification_0729.tif"
      Result = FILE_TEST(result_png_path)
      print,Result
      IF Result EQ 1 THEN begin
        origin_tiff=read_tiff(tif_path, geotiff = geoinfo);
        read_png,result_png_path,result_png
        result_png_rotate=rotate(result_png,7)
        write_tiff,result_tiff_path,result_png_rotate,geotiff=geoinfo
      ENDIF
     
      
    ENDFOR
  ENDFOR
  
END