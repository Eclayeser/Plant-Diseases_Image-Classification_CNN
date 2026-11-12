(function(){"use strict";var H=Object.defineProperty,U=(e,t,o)=>t in e?H(e,t,{enumerable:!0,configurable:!0,writable:!0,value:o}):e[t]=o,I=(e,t,o)=>U(e,typeof t!="symbol"?t+"":t,o);function q(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function _(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}function R(e,t){for(var o="",a=0,s=-1,i=0,r,n=0;n<=e.length;++n){if(n<e.length)r=e.charCodeAt(n);else{if(r===47)break;r=47}if(r===47){if(!(s===n-1||i===1))if(s!==n-1&&i===2){if(o.length<2||a!==2||o.charCodeAt(o.length-1)!==46||o.charCodeAt(o.length-2)!==46){if(o.length>2){var d=o.lastIndexOf("/");if(d!==o.length-1){d===-1?(o="",a=0):(o=o.slice(0,d),a=o.length-1-o.lastIndexOf("/")),s=n,i=0;continue}}else if(o.length===2||o.length===1){o="",a=0,s=n,i=0;continue}}t&&(o.length>0?o+="/..":o="..",a=2)}else o.length>0?o+="/"+e.slice(s+1,n):o=e.slice(s+1,n),a=n-s-1;s=n,i=0}else r===46&&i!==-1?++i:i=-1}return o}function G(e,t){var o=t.dir||t.root,a=t.base||(t.name||"")+(t.ext||"");return o?o===t.root?o+a:o+e+a:a}var b={resolve:function(){for(var e="",t=!1,o,a=arguments.length-1;a>=-1&&!t;a--){var s;a>=0?s=arguments[a]:(o===void 0&&(o=process.cwd()),s=o),_(s),s.length!==0&&(e=s+"/"+e,t=s.charCodeAt(0)===47)}return e=R(e,!t),t?e.length>0?"/"+e:"/":e.length>0?e:"."},normalize:function(e){if(_(e),e.length===0)return".";var t=e.charCodeAt(0)===47,o=e.charCodeAt(e.length-1)===47;return e=R(e,!t),e.length===0&&!t&&(e="."),e.length>0&&o&&(e+="/"),t?"/"+e:e},isAbsolute:function(e){return _(e),e.length>0&&e.charCodeAt(0)===47},join:function(){if(arguments.length===0)return".";for(var e,t=0;t<arguments.length;++t){var o=arguments[t];_(o),o.length>0&&(e===void 0?e=o:e+="/"+o)}return e===void 0?".":b.normalize(e)},relative:function(e,t){if(_(e),_(t),e===t||(e=b.resolve(e),t=b.resolve(t),e===t))return"";for(var o=1;o<e.length&&e.charCodeAt(o)===47;++o);for(var a=e.length,s=a-o,i=1;i<t.length&&t.charCodeAt(i)===47;++i);for(var r=t.length,n=r-i,d=s<n?s:n,c=-1,l=0;l<=d;++l){if(l===d){if(n>d){if(t.charCodeAt(i+l)===47)return t.slice(i+l+1);if(l===0)return t.slice(i+l)}else s>d&&(e.charCodeAt(o+l)===47?c=l:l===0&&(c=0));break}var h=e.charCodeAt(o+l),m=t.charCodeAt(i+l);if(h!==m)break;h===47&&(c=l)}var f="";for(l=o+c+1;l<=a;++l)(l===a||e.charCodeAt(l)===47)&&(f.length===0?f+="..":f+="/..");return f.length>0?f+t.slice(i+c):(i+=c,t.charCodeAt(i)===47&&++i,t.slice(i))},_makeLong:function(e){return e},dirname:function(e){if(_(e),e.length===0)return".";for(var t=e.charCodeAt(0),o=t===47,a=-1,s=!0,i=e.length-1;i>=1;--i)if(t=e.charCodeAt(i),t===47){if(!s){a=i;break}}else s=!1;return a===-1?o?"/":".":o&&a===1?"//":e.slice(0,a)},basename:function(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');_(e);var o=0,a=-1,s=!0,i;if(t!==void 0&&t.length>0&&t.length<=e.length){if(t.length===e.length&&t===e)return"";var r=t.length-1,n=-1;for(i=e.length-1;i>=0;--i){var d=e.charCodeAt(i);if(d===47){if(!s){o=i+1;break}}else n===-1&&(s=!1,n=i+1),r>=0&&(d===t.charCodeAt(r)?--r===-1&&(a=i):(r=-1,a=n))}return o===a?a=n:a===-1&&(a=e.length),e.slice(o,a)}else{for(i=e.length-1;i>=0;--i)if(e.charCodeAt(i)===47){if(!s){o=i+1;break}}else a===-1&&(s=!1,a=i+1);return a===-1?"":e.slice(o,a)}},extname:function(e){_(e);for(var t=-1,o=0,a=-1,s=!0,i=0,r=e.length-1;r>=0;--r){var n=e.charCodeAt(r);if(n===47){if(!s){o=r+1;break}continue}a===-1&&(s=!1,a=r+1),n===46?t===-1?t=r:i!==1&&(i=1):t!==-1&&(i=-1)}return t===-1||a===-1||i===0||i===1&&t===a-1&&t===o+1?"":e.slice(t,a)},format:function(e){if(e===null||typeof e!="object")throw new TypeError('The "pathObject" argument must be of type Object. Received type '+typeof e);return G("/",e)},parse:function(e){_(e);var t={root:"",dir:"",base:"",ext:"",name:""};if(e.length===0)return t;var o=e.charCodeAt(0),a=o===47,s;a?(t.root="/",s=1):s=0;for(var i=-1,r=0,n=-1,d=!0,c=e.length-1,l=0;c>=s;--c){if(o=e.charCodeAt(c),o===47){if(!d){r=c+1;break}continue}n===-1&&(d=!1,n=c+1),o===46?i===-1?i=c:l!==1&&(l=1):i!==-1&&(l=-1)}return i===-1||n===-1||l===0||l===1&&i===n-1&&i===r+1?n!==-1&&(r===0&&a?t.base=t.name=e.slice(1,n):t.base=t.name=e.slice(r,n)):(r===0&&a?(t.name=e.slice(1,i),t.base=e.slice(1,n)):(t.name=e.slice(r,i),t.base=e.slice(r,n)),t.ext=e.slice(i,n)),r>0?t.dir=e.slice(0,r-1):a&&(t.dir="/"),t},sep:"/",delimiter:":",win32:null,posix:null};b.posix=b;var W=b;const S=q(W),j="/home/pyodide",v=e=>`${j}/${e}`,w=(e,t)=>S.resolve(v(e),t);function N(e,t){const o=S.normalize(t),a=S.dirname(o).split("/"),s=[];for(const i of a){s.push(i);const r=s.join("/");if(e.FS.analyzePath(r).exists){if(e.FS.isDir(r))throw new Error(`"${r}" already exists and is not a directory.`);continue}try{e.FS.mkdir(r)}catch(n){throw console.error(`Failed to create a directory "${r}"`),n}}}function P(e,t,o,a){N(e,t),e.FS.writeFile(t,o,a)}function Y(e,t,o){N(e,o),e.FS.rename(t,o)}function z(e){e.forEach(t=>{let o;try{o=new URL(t)}catch{return}if(o.protocol==="emfs:"||o.protocol==="file:")throw new Error(`"emfs:" and "file:" protocols are not allowed for the requirement (${t})`)})}function B(e,t){const o=e.pyimport("packaging.requirements.Requirement");try{const a=o(t);return a.name==="plotly"&&a.specifier.contains("6")}catch{return!1}}function $(e,t){const o=e.pyimport("packaging.requirements.Requirement");try{return o(t).name==="altair"}catch{return!1}}function V(e,t){return t.some(o=>$(e,o))?t.map(o=>B(e,o)?"plotly==5.*":o):t}class J{constructor(){I(this,"_buffer",[]),I(this,"_promise"),I(this,"_resolve"),this._resolve=null,this._promise=null,this._notifyAll()}async _wait(){await this._promise}_notifyAll(){this._resolve&&this._resolve(),this._promise=new Promise(t=>this._resolve=t)}async dequeue(){for(;this._buffer.length===0;)await this._wait();return this._buffer.shift()}enqueue(t){this._buffer.push(t),this._notifyAll()}}function K(e,t,o){const a=new J;o.addEventListener("message",r=>{a.enqueue(r.data)}),o.start();async function s(){return await a.dequeue()}async function i(r){const n=Object.fromEntries(r.toJs());o.postMessage(n)}return e(t,s,i)}const L="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";function X(e){return Array.from(Array(e)).map(()=>L[Math.floor(Math.random()*L.length)]).join("")}const Q=`import jedi


def get_code_completions(code: str, line: int, column: int):
    script = jedi.Script(code)
    completions = script.complete(line, column)
    serializable_completions = [
        {
            "label": completion.name,
            "type": completion.type,
            "docstring": completion.docstring(raw=True),
            "completion_prefix_length": completion.get_completion_prefix_length(),
        }
        for completion in completions
    ]
    return serializable_completions
`;class Z{constructor(t){I(this,"setupPromise"),this.pyodide=t,this.setupPromise=this.setup().catch(o=>(console.error("Error while setting up code completion",o),null))}async setup(){return await this.pyodide.pyimport("micropip").install.callKwargs(["jedi"],{keep_going:!0}),this.pyodide.runPython(Q),this.pyodide.globals.get("get_code_completions")}async getCodeCompletions(t){const o=await this.setupPromise;if(!o)return console.debug("Code completion setup failed, returning empty response"),[];const{code:a,line:s,column:i}=t,r=o(a,s,i),n=r.toJs({dict_converter:Object.fromEntries});return r.destroy(),n}}const ee=`import ast
import os
import sys
import tokenize
import types
from inspect import CO_COROUTINE

from gradio.wasm_utils import app_id_context

# BSD 3-Clause License
#
# - Copyright (c) 2008-Present, IPython Development Team
# - Copyright (c) 2001-2007, Fernando Perez <fernando.perez@colorado.edu>
# - Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>
# - Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Code modified from IPython (BSD license)
# Source: https://github.com/ipython/ipython/blob/master/IPython/utils/syspathcontext.py#L42
class modified_sys_path:  # noqa: N801
    """A context for prepending a directory to sys.path for a second."""

    def __init__(self, script_path: str):
        self._script_path = script_path
        self._added_path = False

    def __enter__(self):
        if self._script_path not in sys.path:
            sys.path.insert(0, self._script_path)
            self._added_path = True

    def __exit__(self, type, value, traceback):
        if self._added_path:
            try:
                sys.path.remove(self._script_path)
            except ValueError:
                # It's already removed.
                pass

        # Returning False causes any exceptions to be re-raised.
        return False


# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# Copyright (c) Yuichiro Tachibana (2023)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def _new_module(name: str) -> types.ModuleType:
    """Create a new module with the given name."""
    return types.ModuleType(name)


def set_home_dir(home_dir: str) -> None:
    os.environ["HOME"] = home_dir
    os.chdir(home_dir)


async def _run_script(app_id: str, home_dir: str, script_path: str) -> None:
    # This function is based on the following code from Streamlit:
    # https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/runtime/scriptrunner/script_runner.py#L519-L554
    # with modifications to support top-level await.
    set_home_dir(home_dir)

    with tokenize.open(script_path) as f:
        filebody = f.read()

    await _run_code(app_id, home_dir, filebody, script_path)


async def _run_code(
        app_id: str,
        home_dir: str,
        filebody: str,
        script_path: str = '<string>'  # This default value follows the convention. Ref: https://docs.python.org/3/library/functions.html#compile
    ) -> None:
    set_home_dir(home_dir)

    # NOTE: In Streamlit, the bytecode caching mechanism has been introduced.
    # However, we skipped it here for simplicity and because Gradio doesn't need to rerun the script so frequently,
    # while we may do it in the future.
    bytecode = compile(  # type: ignore
        filebody,
        # Pass in the file path so it can show up in exceptions.
        script_path,
        # We're compiling entire blocks of Python, so we need "exec"
        # mode (as opposed to "eval" or "single").
        mode="exec",
        # Don't inherit any flags or "future" statements.
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, # Allow top-level await. Ref: https://github.com/whitphx/streamlit/commit/277dc580efb315a3e9296c9a0078c602a0904384
        dont_inherit=1,
        # Use the default optimization options.
        optimize=-1,
    )

    module = _new_module("__main__")

    # Install the fake module as the __main__ module. This allows
    # the pickle module to work inside the user's code, since it now
    # can know the module where the pickled objects stem from.
    # IMPORTANT: This means we can't use "if __name__ == '__main__'" in
    # our code, as it will point to the wrong module!!!
    sys.modules["__main__"] = module

    # Add special variables to the module's globals dict.
    module.__dict__["__file__"] = script_path

    with modified_sys_path(script_path), modified_sys_path(home_dir), app_id_context(app_id):
        # Allow top-level await. Ref: https://github.com/whitphx/streamlit/commit/277dc580efb315a3e9296c9a0078c602a0904384
        if bytecode.co_flags & CO_COROUTINE:
            # The source code includes top-level awaits, so the compiled code object is a coroutine.
            await eval(bytecode, module.__dict__)
        else:
            exec(bytecode, module.__dict__)
`,te=`# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# Copyright (c) Yuichiro Tachibana (2023)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import logging
import os
import sys
import types
from typing import Optional, Set

LOGGER = logging.getLogger(__name__)

#
# Copied from https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/file_util.py
#

def file_is_in_folder_glob(filepath, folderpath_glob) -> bool:
    """Test whether a file is in some folder with globbing support.

    Parameters
    ----------
    filepath : str
        A file path.
    folderpath_glob: str
        A path to a folder that may include globbing.

    """
    # Make the glob always end with "/*" so we match files inside subfolders of
    # folderpath_glob.
    if not folderpath_glob.endswith("*"):
        if folderpath_glob.endswith("/"):
            folderpath_glob += "*"
        else:
            folderpath_glob += "/*"

    file_dir = os.path.dirname(filepath) + "/"
    return fnmatch.fnmatch(file_dir, folderpath_glob)


def get_directory_size(directory: str) -> int:
    """Return the size of a directory in bytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def file_in_pythonpath(filepath) -> bool:
    """Test whether a filepath is in the same folder of a path specified in the PYTHONPATH env variable.


    Parameters
    ----------
    filepath : str
        An absolute file path.

    Returns
    -------
    boolean
        True if contained in PYTHONPATH, False otherwise. False if PYTHONPATH is not defined or empty.

    """
    pythonpath = os.environ.get("PYTHONPATH", "")
    if len(pythonpath) == 0:
        return False

    absolute_paths = [os.path.abspath(path) for path in pythonpath.split(os.pathsep)]
    return any(
        file_is_in_folder_glob(os.path.normpath(filepath), path)
        for path in absolute_paths
    )

#
# Copied from https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/watcher/local_sources_watcher.py
#

def get_module_paths(module: types.ModuleType) -> Set[str]:
    paths_extractors = [
        # https://docs.python.org/3/reference/datamodel.html
        # __file__ is the pathname of the file from which the module was loaded
        # if it was loaded from a file.
        # The __file__ attribute may be missing for certain types of modules
        lambda m: [m.__file__],
        # https://docs.python.org/3/reference/import.html#__spec__
        # The __spec__ attribute is set to the module spec that was used
        # when importing the module. one exception is __main__,
        # where __spec__ is set to None in some cases.
        # https://www.python.org/dev/peps/pep-0451/#id16
        # "origin" in an import context means the system
        # (or resource within a system) from which a module originates
        # ... It is up to the loader to decide on how to interpret
        # and use a module's origin, if at all.
        lambda m: [m.__spec__.origin],
        # https://www.python.org/dev/peps/pep-0420/
        # Handling of "namespace packages" in which the __path__ attribute
        # is a _NamespacePath object with a _path attribute containing
        # the various paths of the package.
        lambda m: list(m.__path__._path),
    ]

    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            # Some modules might not have __file__ or __spec__ attributes.
            pass
        except Exception as e:
            LOGGER.warning(f"Examining the path of {module.__name__} raised: {e}")

        all_paths.update(
            [os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)]
        )
    return all_paths


def _is_valid_path(path: Optional[str]) -> bool:
    return isinstance(path, str) and (os.path.isfile(path) or os.path.isdir(path))


#
# Original code
#

def unload_local_modules(target_dir_path: str = "."):
    """ Unload all modules that are in the target directory or in a subdirectory of it.
    It is necessary to unload modules before re-executing a script that imports the modules,
    so that the new version of the modules is loaded.
    The module unloading feature is extracted from Streamlit's LocalSourcesWatcher (https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/watcher/local_sources_watcher.py)
    and packaged as a standalone function.
    """
    target_dir_path = os.path.abspath(target_dir_path)
    loaded_modules = {} # filepath -> module_name

    # Copied from \`LocalSourcesWatcher.update_watched_modules()\`
    module_paths = {
        name: get_module_paths(module)
        for name, module in dict(sys.modules).items()
    }

    # Copied from \`LocalSourcesWatcher._register_necessary_watchers()\`
    for name, paths in module_paths.items():
        for path in paths:
            if file_is_in_folder_glob(path, target_dir_path) or file_in_pythonpath(path):
                loaded_modules[path] = name

    # Copied from \`LocalSourcesWatcher.on_file_changed()\`
    for module_name in loaded_modules.values():
        if module_name is not None and module_name in sys.modules:
            del sys.modules[module_name]
`;importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.3/full/pyodide.js");let M,x,F,E;async function T(e,t,o,a=3){const s=V(e,o);for(let i=0;i<a;i++){const r=i===a-1;try{return t.install.callKwargs(s,{keep_going:!0})}catch(n){if(r)throw n;console.error("Failed to install packages. Retrying...",n)}}}async function oe(e,t,o,a){console.debug("Loading Pyodide."),t("Loading Pyodide");const s=await loadPyodide({stdout:o,stderr:a});console.debug("Pyodide is loaded."),console.debug("Loading micropip"),t("Loading micropip"),await s.loadPackage("micropip");const i=s.pyimport("micropip");console.debug("micropip is loaded.");const r=[e.gradioWheelUrl,e.gradioClientWheelUrl];console.debug("Loading Gradio wheels.",r),t("Loading Gradio wheels"),await s.loadPackage(["ssl","setuptools"]),await i.add_mock_package("ffmpy","0.3.0"),await T(s,i,["huggingface-hub==0.35.3","anyio==4.11.0","typing_extensions==4.16.0"]),await T(s,i,r),console.debug("Gradio wheels are loaded."),console.debug("Mocking os module methods."),t("Mock os module methods"),await s.runPythonAsync(`
import os

os.link = lambda src, dst: None
`),console.debug("os module methods are mocked."),console.debug("Importing gradio package."),t("Importing gradio package"),await s.runPythonAsync("import gradio"),console.debug("gradio package is imported."),console.debug("Defining a ASGI wrapper function."),t("Defining a ASGI wrapper function"),await s.runPythonAsync(`
# Based on Shiny's App.call_pyodide().
# https://github.com/rstudio/py-shiny/blob/v0.3.3/shiny/_app.py#L224-L258
async def _call_asgi_app_from_js(app_id, scope, receive, send):
	# TODO: Pretty sure there are objects that need to be destroy()'d here?
	scope = scope.to_py()

	# ASGI requires some values to be byte strings, not character strings. Those are
	# not that easy to create in JavaScript, so we let the JS side pass us strings
	# and we convert them to bytes here.
	if "headers" in scope:
			# JS doesn't have \`bytes\` so we pass as strings and convert here
			scope["headers"] = [
					[value.encode("latin-1") for value in header]
					for header in scope["headers"]
			]

	if "query_string" in scope:
		scope["query_string"] = scope["query_string"].encode("latin-1")
	if "raw_path" in scope and scope["raw_path"]:
			scope["raw_path"] = scope["raw_path"].encode("latin-1")

	async def rcv():
			event = await receive()
			py_event = event.to_py()
			if "body" in py_event:
					if isinstance(py_event["body"], memoryview):
							py_event["body"] = py_event["body"].tobytes()
			return py_event

	async def snd(event):
			await send(event)

	app = gradio.wasm_utils.get_registered_app(app_id)
	if app is None:
		raise RuntimeError("Gradio app has not been launched.")

	await app(scope, rcv, snd)
`),M=s.globals.get("_call_asgi_app_from_js"),console.debug("The ASGI wrapper function is defined."),console.debug("Mocking async libraries."),t("Mocking async libraries"),await s.runPythonAsync(`
async def mocked_anyio_to_thread_run_sync(func, *args, cancellable=False, limiter=None):
	return func(*args)

import anyio.to_thread
anyio.to_thread.run_sync = mocked_anyio_to_thread_run_sync
	`),console.debug("Async libraries are mocked."),console.debug("Setting up Python utility functions."),t("Setting up Python utility functions"),await s.runPythonAsync(ee),x=s.globals.get("_run_code"),F=s.globals.get("_run_script"),await s.runPythonAsync(te),E=s.globals.get("unload_local_modules"),console.debug("Python utility functions are set up."),t("Initialization completed");const n=new Z(s);return{pyodide:s,micropip:i,codeCompleter:n}}async function ie(e,t,o,a,s,i){const r=v(o);console.debug("Creating a home directory for the app.",{appId:o,appHomeDir:r}),e.FS.mkdir(r),console.debug("Mounting files.",a.files),s("Mounting files");const n=[];await Promise.all(Object.keys(a.files).map(async m=>{const f=a.files[m];let p;"url"in f?(console.debug(`Fetch a file from ${f.url}`),p=await fetch(f.url).then(y=>y.arrayBuffer()).then(y=>new Uint8Array(y))):p=f.data;const{opts:u}=a.files[m],g=w(o,m);console.debug(`Write a file "${g}"`),P(e,g,p,u),typeof p=="string"&&m.endsWith(".py")&&n.push(p)})),console.debug("Files are mounted."),console.debug("Installing packages.",a.requirements),s("Installing packages"),await T(e,t,a.requirements),console.debug("Packages are installed."),console.debug("Auto-loading modules.");const d=await Promise.all(n.map(m=>e.loadPackagesFromImports(m))),c=new Set(d.flat()),l=Array.from(c);l.length>0&&i(l);const h=l.map(m=>m.name);console.debug("Modules are auto-loaded.",l),(a.requirements.includes("matplotlib")||h.includes("matplotlib"))&&(console.debug("Setting matplotlib backend."),s("Setting matplotlib backend"),await e.runPythonAsync(`
try:
	import matplotlib
	matplotlib.use("agg")
except ImportError:
	pass
`),console.debug("matplotlib backend is set.")),s("App is now loaded")}const O=self;"postMessage"in O?D(O):O.onconnect=e=>{const t=e.ports[0];D(t),t.start()};let A;function D(e){const t=X(8);console.debug("Set up a new app.",{appId:t});const o=d=>{e.postMessage({type:"progress-update",data:{log:d}})},a=d=>{console.log(d),e.postMessage({type:"stdout",data:{output:d}})},s=d=>{console.error(d),e.postMessage({type:"stderr",data:{output:d}})},i=d=>{console.error("Python error:",d),e.postMessage({type:"python-error",data:{traceback:d}})},r=d=>{const c={type:"modules-auto-loaded",data:{packages:d}};e.postMessage(c)};let n;e.onmessage=async function(d){const c=d.data;console.debug("worker.onmessage",c);const l=d.ports[0];try{if(c.type==="init-env"){A==null?A=oe(c.data,o,a,s):o("Pyodide environment initialization is ongoing in another session"),A.then(()=>{const p={type:"reply:success",data:null};l.postMessage(p)}).catch(p=>{const u={type:"reply:error",error:p};l.postMessage(u)});return}if(A==null)throw new Error("Pyodide Initialization is not started.");const{pyodide:h,micropip:m,codeCompleter:f}=await A;if(h.pyimport("gradio").wasm_utils.register_error_traceback_callback(t,i),c.type==="init-app"){n=ie(h,m,t,c.data,o,r);const p={type:"reply:success",data:null};l.postMessage(p);return}if(n==null)throw new Error("App initialization is not started.");switch(await n,c.type){case"echo":{const p={type:"reply:success",data:c.data};l.postMessage(p);break}case"run-python-code":{E(),console.debug("Auto install the requirements");const p=await h.loadPackagesFromImports(c.data.code);p.length>0&&r(p),console.debug("Modules are auto-loaded.",p),await x(t,v(t),c.data.code);const u={type:"reply:success",data:null};l.postMessage(u);break}case"run-python-file":{E(),await F(t,v(t),c.data.path);const p={type:"reply:success",data:null};l.postMessage(p);break}case"asgi-request":{console.debug("ASGI request",c.data),K(M.bind(null,t),c.data.scope,l);break}case"file:write":{const{path:p,data:u,opts:g}=c.data;if(typeof u=="string"&&p.endsWith(".py")){console.debug(`Auto install the requirements in ${p}`);const k=await h.loadPackagesFromImports(u);k.length>0&&r(k),console.debug("Modules are auto-loaded.",k)}const y=w(t,p);console.debug(`Write a file "${y}"`),P(h,y,u,g);const C={type:"reply:success",data:null};l.postMessage(C);break}case"file:rename":{const{oldPath:p,newPath:u}=c.data,g=w(t,p),y=w(t,u);console.debug(`Rename "${g}" to ${y}`),Y(h,g,y);const C={type:"reply:success",data:null};l.postMessage(C);break}case"file:unlink":{const{path:p}=c.data,u=w(t,p);console.debug(`Remove "${u}`),h.FS.unlink(u);const g={type:"reply:success",data:null};l.postMessage(g);break}case"install":{const{requirements:p}=c.data;console.debug("Install the requirements:",p),z(p),await T(h,m,p).then(()=>{if(p.includes("matplotlib"))return h.runPythonAsync(`
try:
	import matplotlib
	matplotlib.use("agg")
except ImportError:
	pass
`)}).then(()=>{console.debug("Successfully installed");const u={type:"reply:success",data:null};l.postMessage(u)});break}case"code-completion":{const p=c.data,u={type:"reply:success",data:await f.getCodeCompletions(p)};l.postMessage(u);break}}}catch(h){if(console.error(h),!(h instanceof Error))throw h;const m=new Error(h.message);m.name=h.name,m.stack=h.stack;const f={type:"reply:error",error:m};l.postMessage(f)}}}})();
//# sourceMappingURL=webworker-YqYS1s-b.js.map
