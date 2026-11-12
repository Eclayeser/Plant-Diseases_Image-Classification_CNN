import{j as r}from"./index-DxSIzuaZ.js";import"./helperFunctions-CjNsMqTA.js";import"./hdrFilteringFunctions-B4VWOPlT.js";import"../lite.js";const e="hdrIrradianceFilteringPixelShader",i=`#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform samplerCube inputTexture;
#ifdef IBL_CDF_FILTERING
uniform sampler2D icdfTexture;
#endif
uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=irradiance(inputTexture,direction,vFilteringInfo
#ifdef IBL_CDF_FILTERING
,icdfTexture
#endif
);gl_FragColor=vec4(color*hdrScale,1.0);}`;r.ShadersStore[e]||(r.ShadersStore[e]=i);const c={name:e,shader:i};export{c as hdrIrradianceFilteringPixelShader};
//# sourceMappingURL=hdrIrradianceFiltering.fragment-4TjDh0xZ.js.map
