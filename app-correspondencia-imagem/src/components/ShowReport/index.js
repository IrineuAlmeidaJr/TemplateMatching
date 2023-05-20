import { useEffect, useState } from "react";
import { useGlobalContext } from "../../context/Global";
import { ModalImageReport } from "../ModalImageReport";

export default function ShowReport() {
    const  globalContext = useGlobalContext();

    const [showImg1, setShowImg1] = useState(true);
    const [showImg2, setShowImg2] = useState(false);
    const [active, setActive] = useState(false);
    const [localImg, setLocalImg] = useState(null);

    function changeImage() {
    }

    function changeButton() {
        console.log("ENTROU")
        if (showImg1) {
            setShowImg1(!showImg1);
            setShowImg2(!showImg2);
            setLocalImg('/images/image_subtraction.jpg')
        }

        if (showImg2) {
            setShowImg1(!showImg1);
            setShowImg2(!showImg2);
            setLocalImg('/images/image_match.jpg')    
        }       
    }

    useEffect(() => {
        setLocalImg('/images/image_match.jpg') 
    }, [])

    return (
        <div
        className="
        flex
        flex-col
        justify-center
        items-center
        m-4
        p-4  
        bg-[#d8ddee] 
        rounded-lg">
            <h2 
            className="
            font-extrabold
            leading-tight 
            tracking-tight 
            text-zinc-800
            text-xl 
            text-center
            uppercase">
                Relatório
            </h2>

            <div className="flex w-full mt-4 mb-4">
                {/* Exibição do Relatório */}
                <div 
                className="
                mr-4 
                mx-2 
                p-4 
                w-1/2
                h-32
                bg-[#ecedf0] 
                rounded-lg"> 
                    <p 
                    className="
                    text-md
                    font-medium 
                    text-zinc-800">
                        <span className="mr-2 font-semibold">
                            Correspondência:
                        </span>
                        {
                            globalContext.report.sucess ? 
                            <span className="text-green-500 font-semibold">
                                sucesso
                            </span>
                        :
                            <span className="text-red-500 font-semibold">
                                erro
                            </span>
                        }
                       
                    </p>

                    <p 
                    className="
                    text-md
                    font-medium 
                    text-zinc-800">
                        <span className="mr-2 font-semibold">
                            Inliers:
                        </span>
                        {globalContext.report.inliers}
                    </p>

                    <p 
                    className="
                    text-md
                    font-medium 
                    text-zinc-800">
                        <span className="mr-2 font-semibold">
                            Outliers:
                        </span>
                        {globalContext.report.outliers}
                    </p>

                    <p 
                    className="
                    text-md
                    font-medium 
                    text-zinc-800">
                        <span className="mr-2 font-semibold">
                            Subtração (img1, img2):
                        </span>
                        { parseFloat(globalContext.report.subtraction.toFixed(2)) }
                    </p>
                </div>

                {/* Exibição da Imagem */}
                <div 
                className=" 
                mx-2 
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
                            Correspondência
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
                            Subtração
                        </button>
                    </div>
                    
                    <div className="bg-[#f6f6f6] p-3 h-[320px] ">
                        <div className="bg-[#f6f6f6] h-[300px] overflow-auto">
                        {
                            localImg != null &&  globalContext.report.sucess ?                
                            <img 
                            src={localImg}
                            alt="imagem a ser comparada"
                            />
                            :
                            <div />
                            
                        }   
                        </div>
                    </div>

                    <div className="flex justify-center">             
                        <button data-modal-target="modal-image" data-modal-toggle="modal-image" 
                        className="
                        w-48
                        mt-4
                        text-white 
                        bg-gray-800 
                        hover:bg-gray-700 
                        focus:outline-none 
                        focus:ring-4 
                        focus:ring-gray-300 
                        font-medium 
                        rounded-lg 
                        text-sm 
                        px-5 
                        py-2.5 
                        mr-2 
                        mb-2 
                        dark:bg-gray-800 
                        dark:hover:bg-gray-700
                        dark:focus:ring-gray-700 
                        dark:border-gray-700
                        uppercase"
                        onClick={() => setActive(true)}
                        >
                            Ampliar Imagem
                        </button>
                    </div>

                    <ModalImageReport 
                    image={localImg} 
                    active={active} 
                    setActive={setActive} />                 
                    
                </div>
            </div>
        </div>
    )   
}