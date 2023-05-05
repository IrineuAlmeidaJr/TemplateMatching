import { createContext, useContext, useReducer, useState } from 'react';

const GlobalContext = createContext(null);

export default function GlobalContextProvider({ children }) {
    const [urlImg1, setUrlImg1] = useState(null);
    const [urlImg2, setUrlImg2] = useState(null);
    const [detector, setDetector] = useState('');
    const [descriptor, setDescriptor] = useState('');

    return (
        <GlobalContext.Provider 
        value={{ 
            urlImg1, 
            setUrlImg1, 
            urlImg2, 
            setUrlImg2, 
            detector, 
            setDetector, 
            descriptor, 
            setDescriptor
        }}>
            { children }
        </GlobalContext.Provider>
    )
}

export const useGlobalContext = () => useContext(GlobalContext);