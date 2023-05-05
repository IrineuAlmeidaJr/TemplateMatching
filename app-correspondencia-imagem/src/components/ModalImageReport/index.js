

export function ModalImageReport({ image, active, setActive }) {

    return (
        <>  
            { active &&
                <div 
                tabIndex="-1" 
                className="
                bg-zinc-500
                bg-opacity-30
                fixed 
                top-0  
                left-0  
                right-0  
                z-50  
                w-full  
                p-4  
                overflow-x-hidden  
                overflow-y-auto 
                h-full">
                    <div className="flex justify-center max-h-full px-10 py-4">
                        <div className="relative px-6 bg-white rounded-lg shadow dark:bg-gray-700">                            
                            <div className="flex items-center justify-between p-2 dark:border-gray-600">
                                <button 
                                onClick={() => setActive(false)}
                                type="button" 
                                className="text-gray-600 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm p-1.5 ml-auto inline-flex items-center dark:hover:bg-gray-600 dark:hover:text-white">
                                    <svg aria-hidden="true" className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path></svg>
                                </button>
                            </div>
                            <div className="pb-6 flex justify-center items-center ">
                                <img 
                                src={image}
                                alt="imagem a ser comparada"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            }
            
        </>
    )
}