import { createContext, useContext, useReducer, useState } from 'react';

const GlobalContext = createContext(null);

export default function GlobalContextProvider({ children }) {
    const [urlImg1, setUrlImg1] = useState(null);
    const [urlImg2, setUrlImg2] = useState(null);
    const [detector, setDetector] = useState('SIFT');
    const [descriptor, setDescriptor] = useState('SIFT');
    const [showReport, setShowReport] = useState(false);
    const [report, setReport] = useState(null);

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
            setDescriptor,
            report,
            setReport,
            showReport,
            setShowReport
        }}>
            { children }
        </GlobalContext.Provider>
    )
}

export const useGlobalContext = () => useContext(GlobalContext);