import { useGlobalContext } from "@/context/Global";
import { useEffect, useState } from "react";
import ModalImage from "../ModalImage";


export default function BoxShowImage() {
    const  globalContext = useGlobalContext();

    const [showImg1, setShowImg1] = useState(true);
    const [showImg2, setShowImg2] = useState(false);
    const [localImg, setLocalImg] = useState(null);

    function changeImage() {
        if (showImg1 && globalContext.urlImg1) {
            setLocalImg(URL.createObjectURL(globalContext.urlImg1))            
        } 

        if (showImg2 && globalContext.urlImg2) {
            setLocalImg(URL.createObjectURL(globalContext.urlImg2))
        } 
    }

    function changeButton() {
        if (showImg1 && globalContext.urlImg2 != null) {
            setShowImg1(!showImg1);
            setShowImg2(!showImg2);
            setLocalImg(URL.createObjectURL(globalContext.urlImg2));
        }

        if (showImg2 && globalContext.urlImg1 != null) {
            setShowImg1(!showImg1);
            setShowImg2(!showImg2);
            setLocalImg(URL.createObjectURL(globalContext.urlImg1)) 
        }
       
    }

    useEffect(() => {
        changeImage();
    },[globalContext.urlImg1, globalContext.urlImg2])

    return (
        <div 
        className="
        mr-4 
        mx-2 
        p-4 
        w-full 
        h-[432px] 
        bg-[#d8ddee] 
        rounded-lg">            
            <div className="flex bg-[#ccd0e4] rounded-lg"> 
                <button 
                className={`
                ${showImg1 ? 
                    'bg-[#f6f6f6] text-zinc-800' 
                    :'bg-[#bfc5da] text-zinc-50'
                }
                flex   
                justify-center
                items-center   
                rounded-tl-lg
                w-1/2
                font-medium 
                text-zinc-800`}
                onClick={changeButton}
                >
                    Image 1
                </button>
                <button
                className={`
                ${showImg2 ? 
                    'bg-[#f6f6f6] text-zinc-800' 
                    :'bg-[#bfc5da] text-zinc-50'
                }
                flex   
                justify-center
                items-center   
                rounded-tr-lg 
                w-1/2
                font-medium `}
                onClick={changeButton}
                >
                    Image 2
                </button>
            </div>
            
            <div className="bg-[#f6f6f6] p-3 h-[320px] ">
                <div className="bg-[#f6f6f6] h-[300px] overflow-auto">
                {
                    localImg != null ?                
                    <img 
                    src={localImg}
                    alt="imagem a ser comparada"
                    />
                    :
                    <div />
                    
                }   
                </div>
            </div>
                
            <ModalImage image={localImg} />         

              


            
        </div>
    )
}