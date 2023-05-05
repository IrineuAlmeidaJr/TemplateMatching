import { useGlobalContext } from "@/context/Global"
import { useRef } from "react";


export default function BoxInput() {    
    const  globalContext = useGlobalContext();
    const buttonRef = useRef(null)

    function handleSubmit(event) {
        event.preventDefault(); 

        
        try {
            const URL = "/api/image-match";
            fetch("/api/image-match", {
                method: 'POST',
                body: JSON.stringify({
                    nameImage1: globalContext.urlImg1.name,
                    nameImage2:  globalContext.urlImg2.name,
                    keypoint: globalContext.detector,
                    descriptor: globalContext.descriptor,
                }),
                headers: {"Content-type": "application/json; charset=UTF-8"}
            })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                globalContext.setReport(data)
            })
            .catch(err => {
                console.log('ERRO NO FETCH -> ' + err)
            });
        } catch (err) {
            console.log(`Erro: ${err}`)
            // Alert.alert('Ops', 'Erro tente novamente');      
        }   
    }

    function handdleShowReport() {
        buttonRef.current.blur(); 
        if (globalContext.report != null) {
            globalContext.setShowReport(!globalContext.showReport)
        }
        
    }

    return (
        <form 
        onSubmit={handleSubmit}
        className="
        ml-4 
        mx-2 
        p-4 
        w-full 
        h-[432px] 
        bg-[#d8ddee] 
        rounded-lg">
            <div className="p-4 bg-[#ccd0e4] rounded-lg">                                
                <label 
                className="
                block 
                mb-1 
                text-sm 
                font-medium 
                text-zinc-800" 
                htmlFor="multiple_files">
                    Enviar 1º imagem
                    <span className="ml-1 font-semibold">
                    formato PNG ou JPG
                    </span>
                </label>
                <input 
                className="
                block 
                w-full 
                text-sm 
                text-gray-900 
                border 
                border-gray-300 
                rounded-lg 
                cursor-pointer 
                bg-gray-50 
                dark:text-gray-400 
                focus:outline-none 
                dark:bg-gray-700 
                dark:border-gray-600 
                dark:placeholder-gray-400"
                id="multiple_files" 
                type="file" 
                accept="image/png, image/jpeg"
                required
                onChange={(e) => globalContext.setUrlImg1(e.target.files[0])}/>
                
                <label 
                className="
                block 
                mt-4
                mb-1 
                text-sm 
                font-medium 
                text-zinc-800" 
                htmlFor="multiple_files">
                    Enviar 2º imagem 
                    <span className="ml-1 font-semibold">
                    formato PNG ou JPG
                    </span>
                </label>
                <input 
                className="
                block 
                w-full 
                text-sm 
                text-gray-900 
                border 
                border-gray-300 
                rounded-lg 
                cursor-pointer 
                bg-gray-50 
                dark:text-gray-400 
                focus:outline-none 
                dark:bg-gray-700 
                dark:border-gray-600 
                dark:placeholder-gray-400"
                id="multiple_files" 
                type="file" 
                accept="image/png, image/jpeg"
                required
                onChange={(e) => globalContext.setUrlImg2(e.target.files[0])}/>
            </div>   
        

            <div className="mt-4">
                <label 
                htmlFor="key_point_detector" 
                className="
                block 
                mb-1 
                text-sm 
                font-medium 
                text-zinc-800">
                    Selecione o detector de ponto chaves 
                </label>
                <select 
                id="key_point_detector" 
                className="
                bg-gray-50 
                border 
                border-gray-300 
                text-gray-900 
                text-sm 
                rounded-lg 
                focus:ring-blue-300 
                focus:border-blue-300 
                block 
                w-full 
                p-2.5 
                dark:bg-gray-700 
                dark:border-gray-600 
                dark:placeholder-gray-400 
                dark:text-white 
                dark:focus:ring-blue-500 
                dark:focus:border-blue-500"
                required
                onChange={(e) => globalContext.setDetector(e.target.value)}
                >
                    <option>Escolha o detector</option>
                    <option value="SIFT">SIFT</option>
                    <option value="SURF">SURF</option>
                    <option value="FAST">FAST</option>
                    <option value="AKAZE">AKAZE</option>
                </select>
            </div>

            <div className="mt-4">
                <label 
                htmlFor="features_descriptor" 
                className="
                block 
                mb-1 
                text-sm 
                font-medium 
                text-zinc-800">
                    Selecione o descritor de características
                </label>
                <select 
                id="features_descriptor" 
                className="
                bg-gray-50 
                border 
                border-gray-300 
                text-gray-900 
                text-sm 
                rounded-lg 
                focus:ring-blue-300 
                focus:border-blue-300 
                block 
                w-full 
                p-2.5 
                dark:bg-gray-700 
                dark:border-gray-600 
                dark:placeholder-gray-400 
                dark:text-white 
                dark:focus:ring-blue-500 
                dark:focus:border-blue-500"
                required
                onChange={(e) => globalContext.setDescriptor(e.target.value)}
                >
                    <option>Escolha o descritor</option>
                    <option value="SIFT">SIFT</option>
                    <option value="SURF">SURF</option>
                    <option value="BRIEF">BRIEF</option>
                    <option value="ORB">ORB</option>
                    <option value="AKAZE">AKAZE</option>
                </select>
            </div>
            
            <div className="mt-4 flex justify-between items-center">
                <button 
                type="submit"
                className   ="
                w-48
                text-white 
                bg-gray-800 
                hover:bg-gray-700 
                focus:text-gray-800
                focus:bg-gray-50
                focus:outline-none 
                focus:ring-4 
                focus:ring-gray-200 
                font-medium 
                rounded-lg 
                text-sm 
                px-5 
                py-2.5 
                mr-2 
                uppercase">
                    enviar
                </button>

                <button 
                type="submit"
                className   ="
                w-48
                text-zinc-800
                bg-gray-50 
                hover:bg-gray-100 
                focus:text-white
                focus:bg-gray-800
                focus:outline-none 
                focus:ring-4 
                focus:ring-gray-200 
                font-medium 
                rounded-lg 
                text-sm 
                px-5 
                py-2.5 
                uppercase">
                    Gerar Relatório
                </button>

                <button 
                type="button" 
                className="
                text-white 
                bg-gray-800 
                hover:bg-gray-700 
                focus:ring-4 
                focus:outline-none 
                focus:text-gray-800
                focus:bg-gray-50
                focus:ring-gray-200 
                font-medium 
                rounded-full 
                text-sm 
                p-2.5 
                text-center 
                inline-flex 
                items-center"
                onClick={handdleShowReport}
                ref={buttonRef}>
                    <svg aria-hidden="true" className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd"></path></svg>
                    <span className="sr-only">Icon to Show</span>
                </button>

            </div>
        </form>
    )
}