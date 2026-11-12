import{j as e}from"./index-DxSIzuaZ.js";import"./helperFunctions-CjNsMqTA.js";import"./hdrFilteringFunctions-B4VWOPlT.js";import"../lite.js";const r="hdrFilteringPixelShader",i=`#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform float alphaG;uniform samplerCube inputTexture;uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=radiance(alphaG,inputTexture,direction,vFilteringInfo);gl_FragColor=vec4(color*hdrScale,1.0);}`;e.ShadersStore[r]||(e.ShadersStore[r]=i);const l={name:r,shader:i};export{l as hdrFilteringPixelShader};
//# sourceMappingURL=hdrFiltering.fragment-D2nF06bm.js.map
